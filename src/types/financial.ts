// 财务数据相关类型定义

export interface TransactionData {
  id: string;
  transactionTime: Date;
  category: string;
  counterparty: string;
  description: string;
  type: '收入' | '支出';
  amount: number;
  paymentMethod: string;
  source: 'wechat' | 'alipay';
}

export interface DataSummary {
  totalIncome: number;
  totalExpense: number;
  netIncome: number;
  transactionCount: number;
  dateRange: {
    start: Date;
    end: Date;
  };
}

export interface ChartData {
  name: string;
  value: number;
  [key: string]: any;
}

export interface CategoryStats {
  category: string;
  amount: number;
  count: number;
  percentage: number;
}

export interface WeeklyData {
  week: string;
  income: number;
  expense: number;
}

export interface FileUploadStatus {
  isUploading: boolean;
  progress: number;
  error: string | null;
  success: boolean;
}