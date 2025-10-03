import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart3, Upload, Table, PieChart, Download, Loader2 } from 'lucide-react';
import FileUpload from '@/components/FileUpload';
import DataOverview from '@/components/DataOverview';
import Charts from '@/components/Charts';
import DataTables from '@/components/DataTables';
import DateRangePicker from '@/components/DateRangePicker';
import { useFinancialStore } from '@/store/financialStore';
import { exportToPDF, ExportData } from '@/utils/pdfExport';
import { toast } from 'sonner';

const Home: React.FC = () => {
  const { transactions, dataSummary, filteredTransactions } = useFinancialStore();
  const [isExporting, setIsExporting] = useState(false);
  const hasData = transactions.length > 0;
  
  // 处理PDF导出
  const handleExportPDF = async () => {
    if (!hasData) {
      toast.error('没有数据可以导出');
      return;
    }
    
    setIsExporting(true);
    
    // 添加重试机制
    const maxRetries = 2;
    let retryCount = 0;
    
    while (retryCount <= maxRetries) {
      try {
        if (retryCount > 0) {
          toast.info(`正在重试导出... (${retryCount}/${maxRetries})`);
          // 等待一段时间让页面完全渲染
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        const activeTransactions = filteredTransactions.length > 0 ? filteredTransactions : transactions;
        const totalIncome = activeTransactions.filter(t => t.type === '收入').reduce((sum, t) => sum + t.amount, 0);
        const totalExpense = activeTransactions.filter(t => t.type === '支出').reduce((sum, t) => sum + t.amount, 0);
        
        // 获取日期范围
        const dates = activeTransactions.map(t => t.transactionTime).sort();
        const startDate = dates[0]?.toLocaleDateString('zh-CN') || '';
        const endDate = dates[dates.length - 1]?.toLocaleDateString('zh-CN') || '';
        const dateRange = startDate === endDate ? startDate : `${startDate} - ${endDate}`;
        
        const exportData: ExportData = {
          totalIncome,
          totalExpense,
          transactionCount: activeTransactions.length,
          dateRange,
          transactions: activeTransactions
        };
        
        await exportToPDF('charts-container', 'tables-container', exportData);
        toast.success('PDF导出成功！');
        break; // 成功则跳出循环
        
      } catch (error) {
        console.error(`PDF导出失败 (尝试 ${retryCount + 1}/${maxRetries + 1}):`, error);
        
        if (retryCount === maxRetries) {
          // 最后一次尝试失败
          const errorMessage = error instanceof Error ? error.message : 'PDF导出失败，请重试';
          toast.error(errorMessage);
        } else {
          retryCount++;
        }
      }
    }
    
    setIsExporting(false);
  };

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-12 max-w-7xl">
        {/* Header Section */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-slate-900 to-slate-700 rounded-2xl mb-6">
            <BarChart3 className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-5xl font-bold tracking-tight text-slate-900 mb-4">
            个人记账系统
          </h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            现代化财务管理平台，让每一笔收支都清晰可见
          </p>
        </div>
        
        <div className="mb-8">
          <FileUpload />
        </div>
        
        {hasData && (
          <div className="space-y-8">
            {/* Date Range Picker */}
            <Card className="border shadow-sm bg-white">
              <CardContent className="p-6">
                <DateRangePicker />
              </CardContent>
            </Card>
            
            {/* Data Overview */}
            <DataOverview />
            
            {/* Charts Section */}
            <Card className="border shadow-sm bg-white">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-2xl font-semibold text-slate-900 flex items-center gap-3">
                    <div className="inline-flex items-center justify-center w-10 h-10 bg-slate-100 rounded-xl">
                      <PieChart className="h-5 w-5 text-slate-600" />
                    </div>
                    数据分析
                  </CardTitle>
                  <Button 
                    onClick={handleExportPDF}
                    disabled={isExporting || !hasData}
                    className="bg-slate-900 hover:bg-slate-800 text-white shadow-sm hover:shadow-md transition-all duration-200"
                    size="lg"
                  >
                    {isExporting ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        导出中...
                      </>
                    ) : (
                      <>
                        <Download className="mr-2 h-4 w-4" />
                        导出PDF
                      </>
                    )}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-6 pt-0">
                <div id="charts-container">
                  <Charts />
                </div>
              </CardContent>
            </Card>
            
            {/* Data Tables Section */}
            <Card className="border shadow-sm bg-white">
              <CardHeader>
                <CardTitle className="text-2xl font-semibold text-slate-900 flex items-center gap-3">
                  <div className="inline-flex items-center justify-center w-10 h-10 bg-slate-100 rounded-xl">
                    <Table className="h-5 w-5 text-slate-600" />
                  </div>
                  数据详情
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 pt-0">
                <div id="tables-container">
                  <DataTables />
                </div>
              </CardContent>
            </Card>
          </div>
        )}


      </div>
    </div>
  );
};

export default Home;