import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer } from 'recharts';
import { useFinancialStore } from '@/store/financialStore';
import { generateCategoryStats, generateWeeklyData } from '@/utils/dataProcessor';
import { format, startOfWeek, endOfWeek } from 'date-fns';

const Charts: React.FC = () => {
  const { transactions } = useFinancialStore();

  if (!transactions || transactions.length === 0) {
    return null;
  }

  // 生成图表数据
  const incomeStats = generateCategoryStats(transactions, '收入');
  const expenseStats = generateCategoryStats(transactions, '支出');
  
  // 重新生成周度数据，确保数据正确显示
  const weeklyData = React.useMemo(() => {
    const weekMap = new Map<string, { income: number; expense: number }>();
    
    transactions.forEach(t => {
      const weekStart = startOfWeek(t.transactionTime, { weekStartsOn: 1 });
      const weekKey = format(weekStart, 'MM-dd');
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
        week,
        income: data.income,
        expense: data.expense
      }))
      .sort((a, b) => a.week.localeCompare(b.week));
  }, [transactions]);

  // 收支对比数据
  const incomeExpenseData = [
    {
      name: '收入',
      value: transactions.filter(t => t.type === '收入').reduce((sum, t) => sum + t.amount, 0),
      fill: '#10b981'
    },
    {
      name: '支出',
      value: transactions.filter(t => t.type === '支出').reduce((sum, t) => sum + t.amount, 0),
      fill: '#ef4444'
    }
  ];

  // 支付方式统计
  const paymentMethodStats = transactions.reduce((acc, t) => {
    const method = t.paymentMethod?.trim() || '未知支付方式';
    // 过滤掉空字符串
    if (method && method !== '未知支付方式') {
      acc[method] = (acc[method] || 0) + t.amount;
    }
    return acc;
  }, {} as Record<string, number>);

  const paymentMethodData = Object.entries(paymentMethodStats)
    .map(([method, amount]) => ({ name: method, value: amount }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8); // 取前8个

  // 调试信息：检查支付方式数据
  console.log('支付方式统计调试信息:', {
    总交易数: transactions.length,
    支付方式样本: transactions.slice(0, 5).map(t => ({ 
      paymentMethod: t.paymentMethod, 
      source: t.source 
    })),
    支付方式统计: paymentMethodStats,
    图表数据: paymentMethodData
  });

  // 简约颜色配置
  const COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', 
    '#8b5cf6', '#06b6d4', '#84cc16', '#f97316',
    '#6366f1', '#14b8a6', '#eab308', '#ec4899'
  ];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 0
    }).format(value);
  };

  return (
    <div className="space-y-6">
      {/* 收支对比饼图 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>收支对比</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* 饼状图 */}
              <div className="lg:col-span-2">
                <ChartContainer
                  config={{
                    income: { label: '收入', color: '#10b981' },
                    expense: { label: '支出', color: '#ef4444' }
                  }}
                  className="h-[280px]"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={incomeExpenseData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={90}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {incomeExpenseData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Pie>
                      <ChartTooltip 
                        content={<ChartTooltipContent formatter={(value) => formatCurrency(Number(value))} />}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
              
              {/* 收支详情列表 */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900 mb-3">收支详情</h4>
                <div className="space-y-2">
                  {incomeExpenseData.map((item, index) => {
                    const total = incomeExpenseData.reduce((sum, d) => sum + d.value, 0);
                    const percentage = total > 0 ? (item.value / total) * 100 : 0;
                    return (
                      <div key={item.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-2">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: item.fill }}
                          ></div>
                          <span className="text-sm font-medium text-gray-700">
                            {item.name}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-semibold text-gray-900">
                            {formatCurrency(item.value)}
                          </div>
                          <div className="text-xs text-gray-500">
                            {percentage.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 周度趋势图 */}
        <Card>
          <CardHeader>
            <CardTitle>周度收支趋势</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                income: { label: '收入', color: '#10b981' },
                expense: { label: '支出', color: '#ef4444' }
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={weeklyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="week" 
                    tick={{ fontSize: 12 }}
                    axisLine={{ stroke: '#e5e7eb' }}
                  />
                  <YAxis 
                    tickFormatter={formatCurrency} 
                    tick={{ fontSize: 12 }}
                    axisLine={{ stroke: '#e5e7eb' }}
                  />
                  <ChartTooltip 
                    content={<ChartTooltipContent formatter={(value) => formatCurrency(Number(value))} />}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="income" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="收入"
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#10b981', strokeWidth: 2 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="expense" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    name="支出"
                    dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#ef4444', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>



      {/* 收入来源分布和支出品类分布 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 收入来源分布 */}
        {incomeStats.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>收入来源分布</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{
                  amount: { label: '金额', color: '#10b981' }
                }}
                className="h-[300px]"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={incomeStats.slice(0, 8)} layout="vertical" barCategoryGap="5%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tickFormatter={formatCurrency} tick={{ fontSize: 12 }} />
                    <YAxis dataKey="category" type="category" width={80} tick={{ fontSize: 12 }} />
                    <ChartTooltip 
                      content={<ChartTooltipContent formatter={(value) => formatCurrency(Number(value))} />}
                    />
                    <Bar dataKey="amount" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            </CardContent>
          </Card>
        )}

        {/* 支出品类分布 */}
        {expenseStats.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>支出品类分布</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{
                  amount: { label: '金额', color: '#ef4444' }
                }}
                className="h-[300px]"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={expenseStats.slice(0, 8)} layout="vertical" barCategoryGap="5%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tickFormatter={formatCurrency} tick={{ fontSize: 12 }} />
                    <YAxis dataKey="category" type="category" width={80} tick={{ fontSize: 12 }} />
                    <ChartTooltip 
                      content={<ChartTooltipContent formatter={(value) => formatCurrency(Number(value))} />}
                    />
                    <Bar dataKey="amount" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            </CardContent>
          </Card>
        )}
      </div>

      {/* 支付方式统计 */}
      {paymentMethodData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>支付方式统计</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                amount: { label: '金额', color: '#6366f1' }
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={paymentMethodData} layout="vertical" barCategoryGap="5%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis type="number" tickFormatter={formatCurrency} tick={{ fontSize: 12 }} />
                  <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                  <ChartTooltip 
                    content={<ChartTooltipContent formatter={(value) => formatCurrency(Number(value))} />}
                  />
                  <Bar dataKey="value" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Charts;