import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Search, Filter, ArrowUpDown } from 'lucide-react';
import { useFinancialStore } from '@/store/financialStore';
import { generateCategoryStats } from '@/utils/dataProcessor';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

type SortField = 'transactionTime' | 'amount' | 'category';
type SortOrder = 'asc' | 'desc';

const DataTables: React.FC = () => {
  const { filteredTransactions } = useFinancialStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<'all' | '收入' | '支出'>('all');
  const [sourceFilter, setSourceFilter] = useState<'all' | 'wechat' | 'alipay'>('all');
  const [sortField, setSortField] = useState<SortField>('transactionTime');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // 过滤和排序数据
  const filteredAndSortedTransactions = useMemo(() => {
    let filtered = filteredTransactions.filter(transaction => {
      const matchesSearch = 
        transaction.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        transaction.counterparty.toLowerCase().includes(searchTerm.toLowerCase()) ||
        transaction.category.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesType = typeFilter === 'all' || transaction.type === typeFilter;
      const matchesSource = sourceFilter === 'all' || transaction.source === sourceFilter;
      
      return matchesSearch && matchesType && matchesSource;
    });

    // 排序
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortField) {
        case 'transactionTime':
          aValue = new Date(a.transactionTime).getTime();
          bValue = new Date(b.transactionTime).getTime();
          break;
        case 'amount':
          aValue = a.amount;
          bValue = b.amount;
          break;
        case 'category':
          aValue = a.category;
          bValue = b.category;
          break;
        default:
          return 0;
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [filteredTransactions, searchTerm, typeFilter, sourceFilter, sortField, sortOrder]);

  // 分页数据
  const paginatedTransactions = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredAndSortedTransactions.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredAndSortedTransactions, currentPage]);

  const totalPages = Math.ceil(filteredAndSortedTransactions.length / itemsPerPage);

  // 统计数据
  const incomeStats = generateCategoryStats(filteredTransactions, '收入');
  const expenseStats = generateCategoryStats(filteredTransactions, '支出');

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 2
    }).format(amount);
  };

  const formatDate = (date: Date) => {
    return format(date, 'yyyy-MM-dd HH:mm', { locale: zhCN });
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  if (!filteredTransactions || filteredTransactions.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        暂无数据可显示
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 统计表格 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 收入分类统计 */}
        {incomeStats.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>收入分类统计</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>分类</TableHead>
                    <TableHead className="text-right">金额</TableHead>
                    <TableHead className="text-right">笔数</TableHead>
                    <TableHead className="text-right">占比</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {incomeStats.slice(0, 10).map((stat, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">{stat.category}</TableCell>
                      <TableCell className="text-right text-green-600">
                        {formatCurrency(stat.amount)}
                      </TableCell>
                      <TableCell className="text-right">{stat.count}</TableCell>
                      <TableCell className="text-right">{stat.percentage.toFixed(1)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {/* 支出分类统计 */}
        {expenseStats.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>支出分类统计</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>分类</TableHead>
                    <TableHead className="text-right">金额</TableHead>
                    <TableHead className="text-right">笔数</TableHead>
                    <TableHead className="text-right">占比</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {expenseStats.slice(0, 10).map((stat, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">{stat.category}</TableCell>
                      <TableCell className="text-right text-red-600">
                        {formatCurrency(stat.amount)}
                      </TableCell>
                      <TableCell className="text-right">{stat.count}</TableCell>
                      <TableCell className="text-right">{stat.percentage.toFixed(1)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </div>

      {/* 交易记录表格 */}
      <Card>
        <CardHeader>
          <CardTitle>交易记录详情</CardTitle>
          <div className="flex flex-col sm:flex-row gap-4">
            {/* 搜索框 */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <Input
                placeholder="搜索交易记录..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            
            {/* 筛选器 */}
            <div className="flex gap-2">
              <Select value={typeFilter} onValueChange={(value: any) => setTypeFilter(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="类型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部</SelectItem>
                  <SelectItem value="收入">收入</SelectItem>
                  <SelectItem value="支出">支出</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={sourceFilter} onValueChange={(value: any) => setSourceFilter(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="来源" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部</SelectItem>
                  <SelectItem value="wechat">微信</SelectItem>
                  <SelectItem value="alipay">支付宝</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>
                  <Button 
                    variant="ghost" 
                    onClick={() => handleSort('transactionTime')}
                    className="h-auto p-0 font-semibold"
                  >
                    交易时间
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>
                  <Button 
                    variant="ghost" 
                    onClick={() => handleSort('category')}
                    className="h-auto p-0 font-semibold"
                  >
                    分类
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>交易对方</TableHead>
                <TableHead>商品说明</TableHead>
                <TableHead>类型</TableHead>
                <TableHead className="text-right">
                  <Button 
                    variant="ghost" 
                    onClick={() => handleSort('amount')}
                    className="h-auto p-0 font-semibold"
                  >
                    金额
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>支付方式</TableHead>
                <TableHead>来源</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedTransactions.map((transaction) => (
                <TableRow key={transaction.id}>
                  <TableCell>{formatDate(transaction.transactionTime)}</TableCell>
                  <TableCell>{transaction.category}</TableCell>
                  <TableCell>{transaction.counterparty}</TableCell>
                  <TableCell className="max-w-xs truncate" title={transaction.description}>
                    {transaction.description}
                  </TableCell>
                  <TableCell>
                    <Badge variant={transaction.type === '收入' ? 'default' : 'destructive'}>
                      {transaction.type}
                    </Badge>
                  </TableCell>
                  <TableCell className={`text-right font-medium ${
                    transaction.type === '收入' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatCurrency(transaction.amount)}
                  </TableCell>
                  <TableCell>{transaction.paymentMethod}</TableCell>
                  <TableCell>
                    <Badge variant="outline">
                      {transaction.source === 'wechat' ? '微信' : '支付宝'}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          
          {/* 分页 */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <div className="text-sm text-gray-500">
                显示 {(currentPage - 1) * itemsPerPage + 1} 到 {Math.min(currentPage * itemsPerPage, filteredAndSortedTransactions.length)} 条，
                共 {filteredAndSortedTransactions.length} 条记录
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                >
                  上一页
                </Button>
                <span className="flex items-center px-3 text-sm">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                >
                  下一页
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default DataTables;