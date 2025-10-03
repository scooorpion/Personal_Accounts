import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, TrendingDown, DollarSign, Calendar } from 'lucide-react';
import { useFinancialStore } from '@/store/financialStore';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

const DataOverview: React.FC = () => {
  const { dataSummary, transactions } = useFinancialStore();

  if (!dataSummary) {
    return null;
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 2
    }).format(amount);
  };

  const formatDate = (date: Date) => {
    return format(date, 'yyyy年MM月dd日', { locale: zhCN });
  };

  const overviewCards = [
    {
      title: '总收入',
      value: formatCurrency(dataSummary.totalIncome),
      icon: TrendingUp,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      description: `共 ${transactions.filter(t => t.type === '收入').length} 笔交易`
    },
    {
      title: '总支出',
      value: formatCurrency(dataSummary.totalExpense),
      icon: TrendingDown,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      description: `共 ${transactions.filter(t => t.type === '支出').length} 笔交易`
    },
    {
      title: '净收入',
      value: formatCurrency(dataSummary.netIncome),
      icon: DollarSign,
      color: dataSummary.netIncome >= 0 ? 'text-green-600' : 'text-red-600',
      bgColor: dataSummary.netIncome >= 0 ? 'bg-green-50' : 'bg-red-50',
      description: dataSummary.netIncome >= 0 ? '盈余' : '亏损'
    },
    {
      title: '交易笔数',
      value: dataSummary.transactionCount.toString(),
      icon: Calendar,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      description: `${formatDate(dataSummary.dateRange.start)} 至 ${formatDate(dataSummary.dateRange.end)}`
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {overviewCards.map((card, index) => {
        const IconComponent = card.icon;
        return (
          <Card key={index} className="hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                {card.title}
              </CardTitle>
              <div className={`p-2 rounded-full ${card.bgColor}`}>
                <IconComponent className={`h-4 w-4 ${card.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                <div className={`text-2xl font-bold ${card.color}`}>
                  {card.value}
                </div>
                <p className="text-xs text-gray-500">
                  {card.description}
                </p>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};

export default DataOverview;