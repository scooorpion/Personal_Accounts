import { create } from 'zustand';
import { TransactionData, DataSummary, FileUploadStatus } from '@/types/financial';
import { processWechatFile, processAlipayFile, generateDataSummary } from '@/utils/dataProcessor';

interface DateRange {
  startDate: Date | null;
  endDate: Date | null;
}

interface FinancialStore {
  // 数据状态
  transactions: TransactionData[];
  filteredTransactions: TransactionData[];
  dataSummary: DataSummary | null;
  
  // 筛选状态
  dateRange: DateRange;
  
  // 上传状态
  uploadStatus: FileUploadStatus;
  
  // 操作方法
  uploadFiles: (files: File[], cumulative?: boolean) => Promise<void>;
  clearData: () => void;
  setUploadStatus: (status: Partial<FileUploadStatus>) => void;
  setDateRange: (dateRange: DateRange) => void;
  applyDateFilter: () => void;
}

export const useFinancialStore = create<FinancialStore>((set, get) => ({
  // 初始状态
  transactions: [],
  filteredTransactions: [],
  dataSummary: null,
  dateRange: {
    startDate: null,
    endDate: null
  },
  uploadStatus: {
    isUploading: false,
    progress: 0,
    error: null,
    success: false
  },
  
  // 上传文件处理
  uploadFiles: async (files: File[], cumulative: boolean = true) => {
    const { setUploadStatus, transactions: existingTransactions } = get();
    
    try {
      setUploadStatus({ 
        isUploading: true, 
        progress: 0, 
        error: null, 
        success: false 
      });
      
      const newTransactions: TransactionData[] = [];
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        
        setUploadStatus({ progress });
        
        let transactions: TransactionData[] = [];
        
        // 根据文件名判断类型
        if (file.name.includes('微信支付') || file.name.includes('wechat')) {
          transactions = await processWechatFile(file);
        } else if (file.name.includes('支付宝') || file.name.includes('alipay')) {
          transactions = await processAlipayFile(file);
        } else {
          // 默认尝试支付宝格式
          transactions = await processAlipayFile(file);
        }
        
        newTransactions.push(...transactions);
      }
      
      // 合并数据：如果是累加模式，合并现有数据；否则替换
      const allTransactions = cumulative 
        ? [...existingTransactions, ...newTransactions]
        : newTransactions;
      
      // 去重：基于交易时间、金额、描述等字段去重
      const uniqueTransactions = allTransactions.filter((transaction, index, self) => {
        return index === self.findIndex(t => 
          t.transactionTime.getTime() === transaction.transactionTime.getTime() &&
          t.amount === transaction.amount &&
          t.description === transaction.description &&
          t.type === transaction.type
        );
      });
      
      // 按时间排序
      uniqueTransactions.sort((a, b) => 
        new Date(b.transactionTime).getTime() - new Date(a.transactionTime).getTime()
      );
      
      // 生成数据摘要
      const dataSummary = generateDataSummary(uniqueTransactions);
      
      set({ 
        transactions: uniqueTransactions,
        filteredTransactions: uniqueTransactions,
        dataSummary 
      });
      
      setUploadStatus({ 
        isUploading: false, 
        progress: 100, 
        success: true 
      });
      
    } catch (error) {
      console.error('文件处理失败:', error);
      setUploadStatus({ 
        isUploading: false, 
        error: error instanceof Error ? error.message : '文件处理失败',
        success: false 
      });
    }
  },
  
  // 清空数据
  clearData: () => {
    set({ 
      transactions: [], 
      filteredTransactions: [],
      dataSummary: null,
      dateRange: {
        startDate: null,
        endDate: null
      },
      uploadStatus: {
        isUploading: false,
        progress: 0,
        error: null,
        success: false
      }
    });
  },
  
  // 设置上传状态
  setUploadStatus: (status: Partial<FileUploadStatus>) => {
    set(state => ({
      uploadStatus: { ...state.uploadStatus, ...status }
    }));
  },
  
  // 设置日期范围
  setDateRange: (dateRange: DateRange) => {
    set({ dateRange });
    get().applyDateFilter();
  },
  
  // 应用日期筛选
  applyDateFilter: () => {
    const { transactions, dateRange } = get();
    
    if (!dateRange.startDate && !dateRange.endDate) {
      // 没有设置日期范围，显示所有数据
      const dataSummary = generateDataSummary(transactions);
      set({ 
        filteredTransactions: transactions,
        dataSummary 
      });
      return;
    }
    
    // 筛选指定日期范围内的交易
    const filtered = transactions.filter(transaction => {
      const transactionDate = new Date(transaction.transactionTime);
      transactionDate.setHours(0, 0, 0, 0);
      
      if (dateRange.startDate && dateRange.endDate) {
        const startDate = new Date(dateRange.startDate);
        const endDate = new Date(dateRange.endDate);
        startDate.setHours(0, 0, 0, 0);
        endDate.setHours(23, 59, 59, 999);
        
        return transactionDate >= startDate && transactionDate <= endDate;
      } else if (dateRange.startDate) {
        const startDate = new Date(dateRange.startDate);
        startDate.setHours(0, 0, 0, 0);
        return transactionDate >= startDate;
      } else if (dateRange.endDate) {
        const endDate = new Date(dateRange.endDate);
        endDate.setHours(23, 59, 59, 999);
        return transactionDate <= endDate;
      }
      
      return true;
    });
    
    // 生成筛选后的数据摘要
    const dataSummary = generateDataSummary(filtered);
    
    set({ 
      filteredTransactions: filtered,
      dataSummary 
    });
  }
}));