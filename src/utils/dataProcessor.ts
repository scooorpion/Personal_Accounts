import * as XLSX from 'xlsx';
import Papa from 'papaparse';
import { format, parseISO } from 'date-fns';
import { TransactionData, DataSummary, CategoryStats, WeeklyData } from '@/types/financial';

// 生成唯一ID
function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

// 查找表头行（支付宝文件）
function findHeaderRow(data: any[][], keywords: string[]): number | null {
  for (let i = 0; i < Math.min(data.length, 30); i++) {
    const rowStr = data[i].join('');
    if (keywords.every(keyword => rowStr.includes(keyword))) {
      return i;
    }
  }
  return null;
}

// 解析微信支付账单
export async function processWechatFile(file: File): Promise<TransactionData[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: 'array' });
        const worksheet = workbook.Sheets[workbook.SheetNames[0]];
        
        // 微信支付账单从第17行开始（跳过前16行的头部信息）
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { 
          header: 1,
          range: 16 // 跳过前16行
        }) as any[][];
        
        if (jsonData.length === 0) {
          resolve([]);
          return;
        }
        
        // 获取表头
        const headers = jsonData[0];
        const transactions: TransactionData[] = [];
        
        // 处理数据行
        for (let i = 1; i < jsonData.length; i++) {
          const row = jsonData[i];
          if (!row || row.length === 0) continue;
          
          const transaction: any = {};
          headers.forEach((header: string, index: number) => {
            transaction[header] = row[index];
          });
          
          // 过滤中性交易（收/支列为"/"的记录）
          if (transaction['收/支'] === '/') continue;
          
          // 数据清理和转换
          const transactionTime = transaction['交易时间'] ? new Date(transaction['交易时间']) : new Date();
          const amount = parseFloat(String(transaction['金额(元)']).replace(/[^\d.-]/g, '')) || 0;
          
          if (isNaN(transactionTime.getTime()) || amount === 0) continue;
          
          const processedTransaction: TransactionData = {
            id: generateId(),
            transactionTime,
            category: transaction['交易类型'] || '其他',
            counterparty: transaction['交易对方'] || '',
            description: transaction['商品'] || '',
            type: transaction['收/支']?.includes('收入') ? '收入' : '支出',
            amount: Math.abs(amount),
            paymentMethod: transaction['支付方式'] || '',
            source: 'wechat'
          };
          
          transactions.push(processedTransaction);
        }
        
        resolve(transactions);
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = () => reject(new Error('文件读取失败'));
    reader.readAsArrayBuffer(file);
  });
}

// 解析支付宝账单
export async function processAlipayFile(file: File): Promise<TransactionData[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        
        if (file.name.endsWith('.csv')) {
          // CSV文件处理
          Papa.parse(text, {
            complete: (results) => {
              try {
                const data = results.data as any[][];
                const headerKeywords = ['交易时间', '交易分类', '商品说明'];
                const headerRowIndex = findHeaderRow(data, headerKeywords);
                
                if (headerRowIndex === null) {
                  reject(new Error('未找到有效的表头'));
                  return;
                }
                
                const headers = data[headerRowIndex];
                const transactions: TransactionData[] = [];
                
                for (let i = headerRowIndex + 1; i < data.length; i++) {
                  const row = data[i];
                  if (!row || row.length === 0) continue;
                  
                  const transaction: any = {};
                  headers.forEach((header: string, index: number) => {
                    transaction[header] = row[index];
                  });
                  
                  // 过滤"不计收支"的记录
                  if (transaction['收/支']?.includes('不计收支')) continue;
                  
                  const transactionTime = transaction['交易时间'] ? new Date(transaction['交易时间']) : new Date();
                  const amount = parseFloat(String(transaction['金额']).replace(/[^\d.-]/g, '')) || 0;
                  
                  if (isNaN(transactionTime.getTime()) || amount === 0) continue;
                  
                  const processedTransaction: TransactionData = {
                    id: generateId(),
                    transactionTime,
                    category: transaction['交易分类'] || '其他',
                    counterparty: transaction['交易对方'] || '',
                    description: transaction['商品说明'] || '',
                    type: transaction['收/支']?.includes('收入') ? '收入' : '支出',
                    amount: Math.abs(amount),
                    paymentMethod: transaction['收/付款方式'] || '',
                    source: 'alipay'
                  };
                  
                  transactions.push(processedTransaction);
                }
                
                resolve(transactions);
              } catch (error) {
                reject(error);
              }
            },
            error: (error) => reject(error)
          });
        } else {
          // Excel文件处理
          const data = new Uint8Array(e.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          const worksheet = workbook.Sheets[workbook.SheetNames[0]];
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as any[][];
          
          const headerKeywords = ['交易时间', '交易分类', '商品说明'];
          const headerRowIndex = findHeaderRow(jsonData, headerKeywords);
          
          if (headerRowIndex === null) {
            reject(new Error('未找到有效的表头'));
            return;
          }
          
          const headers = jsonData[headerRowIndex];
          const transactions: TransactionData[] = [];
          
          for (let i = headerRowIndex + 1; i < jsonData.length; i++) {
            const row = jsonData[i];
            if (!row || row.length === 0) continue;
            
            const transaction: any = {};
            headers.forEach((header: string, index: number) => {
              transaction[header] = row[index];
            });
            
            // 过滤"不计收支"的记录
            if (transaction['收/支']?.includes('不计收支')) continue;
            
            const transactionTime = transaction['交易时间'] ? new Date(transaction['交易时间']) : new Date();
            const amount = parseFloat(String(transaction['金额']).replace(/[^\d.-]/g, '')) || 0;
            
            if (isNaN(transactionTime.getTime()) || amount === 0) continue;
            
            const processedTransaction: TransactionData = {
              id: generateId(),
              transactionTime,
              category: transaction['交易分类'] || '其他',
              counterparty: transaction['交易对方'] || '',
              description: transaction['商品说明'] || '',
              type: transaction['收/支']?.includes('收入') ? '收入' : '支出',
              amount: Math.abs(amount),
              paymentMethod: transaction['收/付款方式'] || '',
              source: 'alipay'
            };
            
            transactions.push(processedTransaction);
          }
          
          resolve(transactions);
        }
      } catch (error) {
        reject(error);
      }
    };
    
    if (file.name.endsWith('.csv')) {
      reader.readAsText(file, 'utf-8');
    } else {
      reader.readAsArrayBuffer(file);
    }
  });
}

// 生成数据摘要
export function generateDataSummary(transactions: TransactionData[]): DataSummary {
  const incomeTransactions = transactions.filter(t => t.type === '收入');
  const expenseTransactions = transactions.filter(t => t.type === '支出');
  
  const totalIncome = incomeTransactions.reduce((sum, t) => sum + t.amount, 0);
  const totalExpense = expenseTransactions.reduce((sum, t) => sum + t.amount, 0);
  
  const dates = transactions.map(t => t.transactionTime).sort();
  
  return {
    totalIncome,
    totalExpense,
    netIncome: totalIncome - totalExpense,
    transactionCount: transactions.length,
    dateRange: {
      start: dates[0] || new Date(),
      end: dates[dates.length - 1] || new Date()
    }
  };
}

// 生成分类统计
export function generateCategoryStats(transactions: TransactionData[], type: '收入' | '支出'): CategoryStats[] {
  const filteredTransactions = transactions.filter(t => t.type === type);
  const categoryMap = new Map<string, { amount: number; count: number }>();
  
  filteredTransactions.forEach(t => {
    const existing = categoryMap.get(t.category) || { amount: 0, count: 0 };
    categoryMap.set(t.category, {
      amount: existing.amount + t.amount,
      count: existing.count + 1
    });
  });
  
  const total = filteredTransactions.reduce((sum, t) => sum + t.amount, 0);
  
  return Array.from(categoryMap.entries())
    .map(([category, stats]) => ({
      category,
      amount: stats.amount,
      count: stats.count,
      percentage: total > 0 ? (stats.amount / total) * 100 : 0
    }))
    .sort((a, b) => b.amount - a.amount);
}

// 生成周度数据
export function generateWeeklyData(transactions: TransactionData[]): WeeklyData[] {
  const weekMap = new Map<string, { income: number; expense: number }>();
  
  transactions.forEach(t => {
    const weekKey = format(t.transactionTime, 'yyyy-ww');
    const existing = weekMap.get(weekKey) || { income: 0, expense: 0 };
    
    if (t.type === '收入') {
      existing.income += t.amount;
    } else {
      existing.expense += t.amount;
    }
    
    weekMap.set(weekKey, existing);
  });
  
  return Array.from(weekMap.entries())
    .map(([week, data]) => ({
      week: `第${week.split('-')[1]}周`,
      income: data.income,
      expense: data.expense
    }))
    .sort((a, b) => a.week.localeCompare(b.week));
}