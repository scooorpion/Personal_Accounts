import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ChartVisualizer:
    def __init__(self, data):
        """
        初始化图表可视化器
        :param data: 处理后的交易数据DataFrame
        """
        self.data = data
        self.prepare_data()
    
    def prepare_data(self):
        """准备和预处理数据"""
        if self.data is None or self.data.empty:
            print("错误：没有可用的数据进行可视化")
            return
        
        # 确保数据类型正确
        if '交易时间' in self.data.columns:
            self.data['交易时间'] = pd.to_datetime(self.data['交易时间'])
            self.data['年月'] = self.data['交易时间'].dt.to_period('M')
        
        # 分离收入和支出
        if '收/支' in self.data.columns:
            self.income_data = self.data[self.data['收/支'].str.contains('收入', na=False)]
            self.expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        else:
            # 如果没有收/支列，根据金额正负判断
            self.income_data = self.data[self.data['金额'] > 0]
            self.expense_data = self.data[self.data['金额'] < 0]
    
    def _plot_income_expense_comparison(self, ax):
        """绘制整体收入支出对比图"""
        # 统计整体收入和支出（融合微信和支付宝数据）
        income_data = self.data[self.data['收/支'].str.contains('收入', na=False)]
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        total_income = income_data['金额'].sum()
        total_expense = abs(expense_data['金额'].sum())
        
        if total_income > 0 or total_expense > 0:
            categories = ['总收入', '总支出']
            amounts = [total_income, total_expense]
            colors = ['#2E8B57', '#DC143C']  # 深绿色和深红色
            
            bars = ax.bar(categories, amounts, color=colors, alpha=0.8)
            ax.set_title('整体收支对比', fontsize=14, fontweight='bold')
            ax.set_ylabel('金额 (元)')
            
            # 添加数值标签
            for bar, amount in zip(bars, amounts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'¥{amount:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 添加净收入信息
            net_income = total_income - total_expense
            net_color = '#2E8B57' if net_income >= 0 else '#DC143C'
            ax.text(0.5, 0.95, f'净收入: ¥{net_income:.2f}', 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, fontweight='bold', color=net_color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def _plot_weekly_trend_subplot(self, ax):
        """在指定轴上绘制周度趋势图"""
        if '交易时间' not in self.data.columns:
            ax.text(0.5, 0.5, '缺少时间数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('周度收入支出趋势')
            return
        
        # 创建周度数据
        data_with_week = self.data.copy()
        data_with_week['年周'] = data_with_week['交易时间'].dt.to_period('W')
        
        # 按周统计收入支出
        income_data_week = data_with_week[data_with_week['收/支'].str.contains('收入', na=False)]
        expense_data_week = data_with_week[data_with_week['收/支'].str.contains('支出', na=False)]
        
        weekly_income = income_data_week.groupby('年周')['金额'].sum() if not income_data_week.empty else pd.Series()
        weekly_expense = expense_data_week.groupby('年周')['金额'].sum().apply(abs) if not expense_data_week.empty else pd.Series()
        
        # 创建完整的周索引
        if not weekly_income.empty or not weekly_expense.empty:
            start_week = min(weekly_income.index.min() if not weekly_income.empty else weekly_expense.index.min(),
                           weekly_expense.index.min() if not weekly_expense.empty else weekly_income.index.min())
            end_week = max(weekly_income.index.max() if not weekly_income.empty else weekly_expense.index.max(),
                         weekly_expense.index.max() if not weekly_expense.empty else weekly_income.index.max())
            
            all_weeks = pd.period_range(start=start_week, end=end_week, freq='W')
            
            weekly_income = weekly_income.reindex(all_weeks, fill_value=0)
            weekly_expense = weekly_expense.reindex(all_weeks, fill_value=0)
        
        if not weekly_income.empty:
            ax.plot(range(len(weekly_income)), weekly_income.values, 
                    marker='o', linewidth=2, label='收入', color='green')
        
        if not weekly_expense.empty:
            ax.plot(range(len(weekly_expense)), weekly_expense.values, 
                    marker='s', linewidth=2, label='支出', color='red')
        
        ax.set_title('周度收入支出趋势')
        ax.set_xlabel('周次')
        ax.set_ylabel('金额')
        ax.legend()
        
        # 设置x轴标签（显示部分周次以避免拥挤）
        if not weekly_income.empty or not weekly_expense.empty:
            week_labels = [f"第{i+1}周" for i in range(len(all_weeks))]
            step = max(1, len(week_labels) // 10)  # 最多显示10个标签
            ax.set_xticks(range(0, len(week_labels), step))
            ax.set_xticklabels([week_labels[i] for i in range(0, len(week_labels), step)], rotation=45)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_income_expense_pie(self, ax):
        """绘制整体收支比例饼图"""
        # 统计整体收入和支出（融合微信和支付宝数据）
        income_data = self.data[self.data['收/支'].str.contains('收入', na=False)]
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        total_income = income_data['金额'].sum()
        total_expense = abs(expense_data['金额'].sum())
        
        if total_income > 0 or total_expense > 0:
            labels = ['收入', '支出']
            sizes = [total_income, total_expense]
            colors = ['#2E8B57', '#DC143C']  # 深绿色和深红色
            explode = (0.05, 0.05)  # 稍微分离饼图片段
            
            # 绘制饼图
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90, 
                                            explode=explode, shadow=True)
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            
            ax.set_title('整体收支比例', fontsize=14, fontweight='bold', pad=20)
            
            # 添加图例，显示具体金额
            legend_labels = [f'{label}: ¥{size:.2f}' for label, size in zip(labels, sizes)]
            ax.legend(wedges, legend_labels, title="详细金额", loc="center left", 
                     bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _plot_expense_category_pie(self, ax):
        """绘制支出品类分布饼图"""
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        if not expense_data.empty:
            # 按交易类型分组统计支出，优先使用'交易分类'列，如果不存在则使用'交易类型'列
            category_column = '交易分类' if '交易分类' in self.data.columns else '交易类型'
            category_stats = expense_data.groupby(category_column)['金额'].sum().abs().sort_values(ascending=False)
            
            if len(category_stats) > 0:
                # 取前8个最大的类别，其余归为"其他"
                if len(category_stats) > 8:
                    top_categories = category_stats.head(8)
                    others_sum = category_stats.tail(len(category_stats) - 8).sum()
                    if others_sum > 0:
                        top_categories['其他'] = others_sum
                    category_stats = top_categories
                
                # 生成颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(category_stats)))
                
                # 绘制饼图
                wedges, texts, autotexts = ax.pie(category_stats.values, 
                                                labels=category_stats.index, 
                                                colors=colors,
                                                autopct='%1.1f%%', 
                                                startangle=90,
                                                shadow=True)
                
                # 美化文本
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                for text in texts:
                    text.set_fontsize(10)
                
                ax.set_title('支出品类分布', fontsize=14, fontweight='bold', pad=20)
                
                # 添加图例，显示具体金额
                legend_labels = [f'{cat}: ¥{amount:.2f}' for cat, amount in category_stats.items()]
                ax.legend(wedges, legend_labels, title="详细金额", loc="center left", 
                         bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
            else:
                ax.text(0.5, 0.5, '暂无支出数据', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, '暂无支出数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)

    def _plot_income_source_pie(self, ax):
        """绘制收入来源分布饼图"""
        income_data = self.data[self.data['收/支'].str.contains('收入', na=False)]
        
        if not income_data.empty:
            # 按交易类型分组统计收入，优先使用'交易分类'列，如果不存在则使用'交易类型'列
            category_column = '交易分类' if '交易分类' in self.data.columns else '交易类型'
            source_stats = income_data.groupby(category_column)['金额'].sum().sort_values(ascending=False)
            
            if len(source_stats) > 0:
                # 取前8个最大的来源，其余归为"其他"
                if len(source_stats) > 8:
                    top_sources = source_stats.head(8)
                    others_sum = source_stats.tail(len(source_stats) - 8).sum()
                    if others_sum > 0:
                        top_sources['其他'] = others_sum
                    source_stats = top_sources
                
                # 生成颜色
                colors = plt.cm.Set2(np.linspace(0, 1, len(source_stats)))
                
                # 绘制饼图
                wedges, texts, autotexts = ax.pie(source_stats.values, 
                                                labels=source_stats.index, 
                                                colors=colors,
                                                autopct='%1.1f%%', 
                                                startangle=90,
                                                shadow=True)
                
                # 美化文本
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                for text in texts:
                    text.set_fontsize(10)
                
                ax.set_title('收入来源分布', fontsize=14, fontweight='bold', pad=20)
                
                # 添加图例，显示具体金额
                legend_labels = [f'{source}: ¥{amount:.2f}' for source, amount in source_stats.items()]
                ax.legend(wedges, legend_labels, title="详细金额", loc="center left", 
                         bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
            else:
                ax.text(0.5, 0.5, '暂无收入数据', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, '暂无收入数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_payment_method_subplot(self, ax):
        """绘制消费分类分析（使用真正的商品类别）"""
        # 统计所有支出交易的商品分类
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)].copy()
        
        if expense_data.empty:
            ax.text(0.5, 0.5, '暂无支出数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('消费分类分析')
            return
        
        # 统一分类字段：微信使用"交易类型"，支付宝使用"交易分类"
        expense_data['统一分类'] = expense_data.apply(lambda row: 
            row.get('交易类型', row.get('交易分类', row.get('商品说明', '其他'))), axis=1)
        
        # 按分类统计支出金额（使用统一分类字段）
        category_stats = expense_data.groupby('统一分类').agg({
            '金额': ['sum', 'count']
        }).round(2)
        
        category_stats.columns = ['总金额', '交易次数']
        category_stats['总金额'] = category_stats['总金额'].abs()  # 取绝对值
        category_stats = category_stats.sort_values('总金额', ascending=True)
        
        # 取前8个分类
        top_categories = category_stats.tail(8)
        
        if top_categories.empty:
            ax.text(0.5, 0.5, '暂无分类数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('消费分类分析')
            return
        
        # 绘制水平条形图
        categories = top_categories.index
        amounts = top_categories['总金额']
        counts = top_categories['交易次数']
        
        # 使用渐变色彩
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        bars = ax.barh(categories, amounts, color=colors, alpha=0.8)
        ax.set_title('消费分类分析 (TOP8)', fontsize=14, fontweight='bold')
        ax.set_xlabel('消费金额 (元)')
        
        # 添加数值标签
        for i, (category, amount, count) in enumerate(zip(categories, amounts, counts)):
            ax.text(amount + max(amounts) * 0.01, i, 
                   f'¥{amount:.0f} ({count}笔)', 
                   va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        
        # 调整布局
        plt.setp(ax.get_yticklabels(), fontsize=10)
    
    def _plot_income_source_analysis(self, ax):
        """绘制收入来源详细分析表格"""
        # 只分析收入数据
        income_data = self.data[self.data['收/支'].str.contains('收入', na=False)].copy()
        
        if income_data.empty:
            ax.text(0.5, 0.5, '暂无收入数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('收入来源分析')
            ax.axis('off')
            return
        
        # 统一分类字段：微信使用"交易类型"，支付宝使用"交易分类"
        income_data['收入类型'] = income_data.apply(lambda row: 
            row.get('交易类型', row.get('交易分类', '其他')), axis=1)
        
        # 按收入类型统计
        income_stats = income_data.groupby('收入类型').agg({
            '金额': ['sum', 'count', 'mean']
        }).round(2)
        
        income_stats.columns = ['总金额', '交易次数', '平均金额']
        income_stats = income_stats.sort_values('总金额', ascending=False)
        
        # 计算占比
        total_income = income_stats['总金额'].sum()
        income_stats['占比'] = (income_stats['总金额'] / total_income * 100).round(1)
        
        # 创建表格数据
        table_data = []
        for idx, (category, row) in enumerate(income_stats.iterrows()):
            table_data.append([
                category,
                f'¥{row["总金额"]:.2f}',
                f'{row["交易次数"]:.0f}笔',
                f'¥{row["平均金额"]:.2f}',
                f'{row["占比"]:.1f}%'
            ])
        
        # 添加总计行
        table_data.append([
            '总计',
            f'¥{total_income:.2f}',
            f'{income_stats["交易次数"].sum():.0f}笔',
            f'¥{income_stats["总金额"].mean():.2f}',
            '100.0%'
        ])
        
        # 创建表格
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=['收入类型', '总金额', '交易次数', '平均金额', '占比'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置总计行样式
        for i in range(5):
            table[(len(table_data), i)].set_facecolor('#E8F5E8')
            table[(len(table_data), i)].set_text_props(weight='bold')
        
        # 设置数据行颜色
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
        
        ax.set_title('收入来源详细分析', fontsize=14, fontweight='bold', pad=20)

    def _plot_summary_stats(self, ax):
        """绘制统计摘要表格"""
        # 生成统计数据
        stats = self._generate_summary_data()
        
        # 创建表格数据
        table_data = [
            ['总交易笔数', f"{stats['total_transactions']}笔"],
            ['总收入', f"¥{stats['total_income']:.2f}"],
            ['总支出', f"¥{stats['total_expense']:.2f}"],
            ['净收入', f"¥{stats['net_income']:.2f}"],
            ['', ''],
            ['收入笔数', f"{stats['income_count']}笔"],
            ['支出笔数', f"{stats['expense_count']}笔"],
            ['', ''],
            ['平均收入', f"¥{stats['avg_income']:.2f}"],
            ['平均支出', f"¥{stats['avg_expense']:.2f}"],
        ]
        
        # 创建表格
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=['统计项目', '数值'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置重要数据行样式
        important_rows = [1, 2, 3, 4]  # 总收入、总支出、净收入行
        for row in important_rows:
            for col in range(2):
                table[(row, col)].set_facecolor('#E3F2FD')
                table[(row, col)].set_text_props(weight='bold')
        
        # 设置净收入行颜色（根据盈亏）
        net_color = '#C8E6C9' if stats['net_income'] >= 0 else '#FFCDD2'
        for col in range(2):
            table[(4, col)].set_facecolor(net_color)
            table[(4, col)].set_text_props(weight='bold')
        
        # 设置分隔行
        separator_rows = [5, 8]
        for row in separator_rows:
            for col in range(2):
                table[(row, col)].set_facecolor('#F5F5F5')
        
        ax.set_title('财务数据统计摘要', fontsize=14, fontweight='bold', pad=20)
    
    def _generate_summary_data(self):
        """生成统计摘要数据"""
        total_transactions = len(self.data)
        income_data = self.data[self.data['收/支'].str.contains('收入', na=False)]
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        total_income = income_data['金额'].sum() if not income_data.empty else 0
        total_expense = abs(expense_data['金额'].sum()) if not expense_data.empty else 0
        net_income = total_income - total_expense
        
        avg_income = income_data['金额'].mean() if not income_data.empty else 0
        avg_expense = abs(expense_data['金额'].mean()) if not expense_data.empty else 0
        
        return {
            'total_transactions': total_transactions,
            'total_income': total_income,
            'total_expense': total_expense,
            'net_income': net_income,
            'income_count': len(income_data),
            'expense_count': len(expense_data),
            'avg_income': avg_income,
            'avg_expense': avg_expense
        }

    def _plot_monthly_summary_table(self, ax):
        """绘制月度统计摘要表格"""
        # 确保日期列是datetime类型
        if '交易时间' in self.data.columns:
            self.data['交易时间'] = pd.to_datetime(self.data['交易时间'])
            
            # 按月分组统计
            monthly_stats = []
            for month, group in self.data.groupby(self.data['交易时间'].dt.to_period('M')):
                income_data = group[group['收/支'].str.contains('收入', na=False)]
                expense_data = group[group['收/支'].str.contains('支出', na=False)]
                
                total_income = income_data['金额'].sum()
                total_expense = abs(expense_data['金额'].sum())
                net_income = total_income - total_expense
                
                monthly_stats.append([
                    str(month),
                    f"{len(group)}笔",
                    f"¥{total_income:.2f}",
                    f"¥{total_expense:.2f}",
                    f"¥{net_income:.2f}"
                ])
            
            # 如果没有月度数据，创建总体统计
            if not monthly_stats:
                stats = self._generate_summary_data()
                monthly_stats = [[
                    "总计",
                    f"{stats['total_transactions']}笔",
                    f"¥{stats['total_income']:.2f}",
                    f"¥{stats['total_expense']:.2f}",
                    f"¥{stats['net_income']:.2f}"
                ]]
        else:
            # 如果没有时间数据，显示总体统计
            stats = self._generate_summary_data()
            monthly_stats = [[
                "总计",
                f"{stats['total_transactions']}笔",
                f"¥{stats['total_income']:.2f}",
                f"¥{stats['total_expense']:.2f}",
                f"¥{stats['net_income']:.2f}"
            ]]
        
        # 创建表格
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=monthly_stats,
                        colLabels=['月份', '交易笔数', '总收入', '总支出', '净收入'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(monthly_stats) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                    
                # 净收入列根据盈亏设置颜色
                if j == 4:  # 净收入列
                    net_value = float(monthly_stats[i-1][4].replace('¥', '').replace(',', ''))
                    color = '#C8E6C9' if net_value >= 0 else '#FFCDD2'
                    table[(i, j)].set_facecolor(color)
        
        ax.set_title('月度财务统计摘要', fontsize=14, fontweight='bold', pad=20)

    def _plot_weekly_spending_pattern(self, ax):
        """绘制一周消费习惯分析"""
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        if expense_data.empty:
            ax.text(0.5, 0.5, '暂无支出数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('一周消费习惯分析')
            return
        
        # 添加星期列
        expense_data = expense_data.copy()
        expense_data['星期'] = expense_data['交易时间'].dt.dayofweek
        
        # 按星期统计支出
        weekly_stats = expense_data.groupby('星期').agg({
            '金额': ['sum', 'count']
        }).round(2)
        
        weekly_stats.columns = ['总金额', '交易次数']
        weekly_stats['总金额'] = weekly_stats['总金额'].abs()
        
        if weekly_stats.empty:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('一周消费习惯分析')
            return
        
        # 星期标签
        weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        # 创建完整的星期数据
        full_week = pd.DataFrame(index=range(7))
        weekly_stats = full_week.join(weekly_stats, how='left').fillna(0)
        
        # 绘制柱状图
        colors = ['#FF6B6B' if i < 5 else '#4ECDC4' for i in range(7)]  # 工作日红色，周末蓝绿色
        bars = ax.bar(range(7), weekly_stats['总金额'], color=colors, alpha=0.8)
        
        ax.set_title('一周消费习惯分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('消费金额 (元)')
        ax.set_xlabel('星期')
        ax.set_xticks(range(7))
        ax.set_xticklabels(weekday_labels)
        
        # 添加数值标签
        for i, (amount, count) in enumerate(zip(weekly_stats['总金额'], weekly_stats['交易次数'])):
            if amount > 0:
                ax.text(i, amount + weekly_stats['总金额'].max() * 0.01,
                       f'¥{amount:.0f}\n({int(count)}笔)', ha='center', va='bottom', fontsize=9)
        
        # 添加工作日/周末平均线
        weekday_avg = weekly_stats['总金额'][:5].mean()
        weekend_avg = weekly_stats['总金额'][5:].mean()
        
        ax.axhline(y=weekday_avg, color='red', linestyle='--', alpha=0.7, label=f'工作日均值: ¥{weekday_avg:.0f}')
        ax.axhline(y=weekend_avg, color='blue', linestyle='--', alpha=0.7, label=f'周末均值: ¥{weekend_avg:.0f}')
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_hourly_spending_pattern(self, ax):
        """绘制一天消费时段分析"""
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        if expense_data.empty:
            ax.text(0.5, 0.5, '暂无支出数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('一天消费时段分析')
            return
        
        # 添加小时列
        expense_data = expense_data.copy()
        expense_data['小时'] = expense_data['交易时间'].dt.hour
        
        # 按小时统计支出
        hourly_stats = expense_data.groupby('小时').agg({
            '金额': ['sum', 'count']
        }).round(2)
        
        hourly_stats.columns = ['总金额', '交易次数']
        hourly_stats['总金额'] = hourly_stats['总金额'].abs()
        
        if hourly_stats.empty:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('一天消费时段分析')
            return
        
        # 绘制折线图
        ax.plot(hourly_stats.index, hourly_stats['总金额'], 
               marker='o', linewidth=2, markersize=6, color='#FF6B6B', alpha=0.8)
        
        ax.set_title('一天消费时段分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('消费金额 (元)')
        ax.set_xlabel('小时')
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
        
        # 标注峰值时段
        max_hour = hourly_stats['总金额'].idxmax()
        max_amount = hourly_stats['总金额'].max()
        ax.annotate(f'峰值: {max_hour}时\n¥{max_amount:.0f}', 
                   xy=(max_hour, max_amount), xytext=(max_hour+2, max_amount*1.1),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.grid(True, alpha=0.3)

    def _plot_top_merchants_analysis(self, ax):
        """绘制主要消费商户分析"""
        expense_data = self.data[self.data['收/支'].str.contains('支出', na=False)]
        
        if expense_data.empty:
            ax.text(0.5, 0.5, '暂无支出数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('主要消费商户分析')
            return
        
        # 按交易对方统计支出
        merchant_stats = expense_data.groupby('交易对方').agg({
            '金额': ['sum', 'count']
        }).round(2)
        
        merchant_stats.columns = ['总金额', '交易次数']
        merchant_stats['总金额'] = merchant_stats['总金额'].abs()
        merchant_stats = merchant_stats.sort_values('总金额', ascending=True)
        
        # 取前10个商户
        top_merchants = merchant_stats.tail(10)
        
        if top_merchants.empty:
            ax.text(0.5, 0.5, '暂无商户数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('主要消费商户分析')
            return
        
        # 绘制水平条形图
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_merchants)))
        bars = ax.barh(top_merchants.index, top_merchants['总金额'], color=colors, alpha=0.8)
        
        ax.set_title('主要消费商户分析 (TOP10)', fontsize=14, fontweight='bold')
        ax.set_xlabel('消费金额 (元)')
        
        # 添加数值标签
        for bar, amount, count in zip(bars, top_merchants['总金额'], top_merchants['交易次数']):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'¥{amount:.0f} ({int(count)}次)', 
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.setp(ax.get_yticklabels(), fontsize=9)

    def generate_summary_statistics(self):
        """生成收入支出统计摘要"""
        print("=" * 50)
        print("个人财务数据统计摘要")
        print("=" * 50)
        
        if self.data is None or self.data.empty:
            print("暂无数据可供分析")
            return
        
        # 基本统计
        total_transactions = len(self.data)
        income_total = self.income_data['金额'].sum() if not self.income_data.empty else 0
        expense_total = abs(self.expense_data['金额'].sum()) if not self.expense_data.empty else 0
        net_income = income_total - expense_total
        
        print(f"总交易笔数: {total_transactions}")
        print(f"总收入: ¥{income_total:.2f}")
        print(f"总支出: ¥{expense_total:.2f}")
        print(f"净收入: ¥{net_income:.2f}")
        print()
        
        # 收入分类统计
        if not self.income_data.empty:
            print("收入分类 TOP5:")
            income_categories = self.income_data.groupby('商品说明')['金额'].sum().sort_values(ascending=False).head(5)
            for category, amount in income_categories.items():
                print(f"  {category}: ¥{amount:.2f}")
        print()
        
        # 支出分类统计
        if not self.expense_data.empty:
            print("支出分类 TOP5:")
            expense_categories = self.expense_data.groupby('商品说明')['金额'].sum().apply(abs).sort_values(ascending=False).head(5)
            for category, amount in expense_categories.items():
                print(f"  {category}: ¥{amount:.2f}")
        print()
        
        # 支付方式统计
        if '收/付款方式' in self.data.columns:
            print("支付方式统计:")
            payment_methods = self.data.groupby('收/付款方式')['金额'].sum().apply(abs).sort_values(ascending=False)
            for method, amount in payment_methods.items():
                print(f"  {method}: ¥{amount:.2f}")
        
        print("=" * 50)
    
    def _create_combined_detail_table(self, fig):
        """创建合并的详细表格，紧凑显示在一页内"""
        # 清除图形
        fig.clear()
        
        # 创建两个子图：支出和收入
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        # 支出分类汇总表
        expense_data = self.data[self.data['金额'] < 0].copy()
        if not expense_data.empty:
            # 使用正确的列名
            category_col = '交易分类' if '交易分类' in expense_data.columns else '交易类型'
            expense_summary = expense_data.groupby(category_col).agg({
                '金额': ['sum', 'count'],
                '交易对方': lambda x: ', '.join(x.unique()[:3])  # 只显示前3个
            }).round(2)
            
            # 重命名列
            expense_summary.columns = ['总金额', '交易次数', '主要商户']
            expense_summary['总金额'] = expense_summary['总金额'].abs()  # 转为正数显示
            expense_summary = expense_summary.sort_values('总金额', ascending=False).head(10)  # 只显示前10项
            
            # 创建表格
            ax1.axis('tight')
            ax1.axis('off')
            ax1.set_title('支出分类汇总 (前10项)', fontsize=12, fontweight='bold', pad=10)
            
            table1 = ax1.table(cellText=expense_summary.values,
                              rowLabels=expense_summary.index,
                              colLabels=expense_summary.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
            
            # 设置表格样式
            table1.auto_set_font_size(False)
            table1.set_fontsize(8)
            table1.scale(1, 1.2)
            
            # 设置表头样式
            for i in range(len(expense_summary.columns)):
                table1[(0, i)].set_facecolor('#4CAF50')
                table1[(0, i)].set_text_props(weight='bold', color='white')
        
        # 收入来源汇总表
        income_data = self.data[self.data['金额'] > 0].copy()
        if not income_data.empty:
            # 使用正确的列名
            category_col = '交易分类' if '交易分类' in income_data.columns else '交易类型'
            income_summary = income_data.groupby(category_col).agg({
                '金额': ['sum', 'count'],
                '交易对方': lambda x: ', '.join(x.unique()[:3])  # 只显示前3个
            }).round(2)
            
            # 重命名列
            income_summary.columns = ['总金额', '交易次数', '主要来源']
            income_summary = income_summary.sort_values('总金额', ascending=False).head(10)  # 只显示前10项
            
            # 创建表格
            ax2.axis('tight')
            ax2.axis('off')
            ax2.set_title('收入来源汇总 (前10项)', fontsize=12, fontweight='bold', pad=10)
            
            table2 = ax2.table(cellText=income_summary.values,
                              rowLabels=income_summary.index,
                              colLabels=income_summary.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
            
            # 设置表格样式
            table2.auto_set_font_size(False)
            table2.set_fontsize(8)
            table2.scale(1, 1.2)
            
            # 设置表头样式
            for i in range(len(income_summary.columns)):
                table2[(0, i)].set_facecolor('#2196F3')
                table2[(0, i)].set_text_props(weight='bold', color='white')
        
        # 调整子图间距
        plt.subplots_adjust(hspace=0.4)
        """为每个消费品类创建详细表格页面"""
        if self.expense_data.empty:
            return []
        
        # 按交易分类分组
        category_groups = self.expense_data.groupby('交易分类')
        detail_pages = []
        
        for category, group_data in category_groups:
            if group_data.empty:
                continue
                
            # 创建新的图形页面
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # 准备表格数据
            table_data = []
            table_data.append(['商品说明', '金额(元)', '交易时间', '支付方式'])
            
            # 按金额降序排列
            sorted_data = group_data.sort_values('金额', key=abs, ascending=False)
            
            for _, row in sorted_data.iterrows():
                amount_str = f"{abs(row['金额']):.2f}"
                time_str = row['交易时间'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['交易时间']) else '未知'
                payment_method = row.get('收/付款方式', '未知')
                description = row.get('商品说明', '未知')
                
                table_data.append([description, amount_str, time_str, payment_method])
            
            # 创建表格
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center')
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # 设置表头样式
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 设置数据行样式（交替颜色）
            for i in range(1, len(table_data)):
                for j in range(len(table_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            # 设置标题
            total_amount = abs(group_data['金额'].sum())
            transaction_count = len(group_data)
            ax.set_title(f'{category} - 详细消费记录\n'
                        f'总金额: ¥{total_amount:.2f} | 交易笔数: {transaction_count}笔',
                        fontsize=14, fontweight='bold', pad=20)
            
            detail_pages.append(fig)
        
        return detail_pages
    
    def _create_income_source_detail_tables(self):
        """为每个收入来源创建详细表格页面"""
        if self.income_data.empty:
            return []
        
        # 按交易分类分组
        category_groups = self.income_data.groupby('交易分类')
        detail_pages = []
        
        for category, group_data in category_groups:
            if group_data.empty:
                continue
                
            # 创建新的图形页面
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # 准备表格数据
            table_data = []
            table_data.append(['收入说明', '金额(元)', '交易时间', '收款方式'])
            
            # 按金额降序排列
            sorted_data = group_data.sort_values('金额', ascending=False)
            
            for _, row in sorted_data.iterrows():
                amount_str = f"{row['金额']:.2f}"
                time_str = row['交易时间'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['交易时间']) else '未知'
                payment_method = row.get('收/付款方式', '未知')
                description = row.get('商品说明', '未知')
                
                table_data.append([description, amount_str, time_str, payment_method])
            
            # 创建表格
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center')
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # 设置表头样式
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 设置数据行样式（交替颜色）
            for i in range(1, len(table_data)):
                for j in range(len(table_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            # 设置标题
            total_amount = group_data['金额'].sum()
            transaction_count = len(group_data)
            ax.set_title(f'{category} - 详细收入记录\n'
                        f'总金额: ¥{total_amount:.2f} | 交易笔数: {transaction_count}笔',
                        fontsize=14, fontweight='bold', pad=20)
            
            detail_pages.append(fig)
        
        return detail_pages

    def show_all_charts(self):
        """显示所有图表（带滚动功能）"""
        # 创建更大的图形以容纳所有图表
        fig = plt.figure(figsize=(20, 20))
        
        # 调整子图布局 - 现在有6个图表
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2, 
                             top=0.95, bottom=0.05, left=0.08, right=0.92)
        
        # 创建子图
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        
        # 绘制各个图表
        self._plot_income_expense_comparison(ax1)
        self._plot_weekly_trend_subplot(ax2)
        self._plot_payment_method_subplot(ax3)
        self._plot_income_source_analysis(ax4)
        self._plot_monthly_summary_table(ax5)
        self._plot_summary_stats(ax6)
        
        # 设置整体标题
        fig.suptitle('个人财务数据分析报告', fontsize=20, fontweight='bold', y=0.98)
        
        # 尝试设置窗口大小和滚动功能
        try:
            mngr = plt.get_current_fig_manager()
            if hasattr(mngr, 'window') and mngr.window:
                mngr.window.wm_geometry("1400x900+100+50")  # 设置窗口大小和位置
        except Exception as e:
            print(f"无法设置窗口属性: {e}")
        
        plt.show()
        return fig
    
    def export_to_pdf(self, filename=None):
        """导出图表为PDF文件，优化布局和页面数量"""
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，不显示图表
        
        # 自动生成文件名
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"财务分析报告_{timestamp}.pdf"
        
        print(f"正在导出PDF文件: {filename}")
        
        # 统一页面大小
        page_size = (16, 20)
        
        with PdfPages(filename) as pdf:
            # 第一页：主要图表概览
            fig = plt.figure(figsize=page_size)
            fig.suptitle('个人财务分析报告', fontsize=18, fontweight='bold', y=0.98)
            
            # 调整子图间距
            plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95, bottom=0.05)
            
            # 子图1: 收入支出对比
            ax1 = plt.subplot(3, 2, 1)
            self._plot_income_expense_comparison(ax1)
            
            # 子图2: 周度趋势
            ax2 = plt.subplot(3, 2, 2)
            self._plot_weekly_trend_subplot(ax2)
            
            # 子图3: 收入支出饼图
            ax3 = plt.subplot(3, 2, 3)
            self._plot_income_expense_pie(ax3)
            
            # 子图4: 收入来源分析
            ax4 = plt.subplot(3, 2, 4)
            self._plot_income_source_analysis(ax4)
            
            # 子图5: 月度统计总览
            ax5 = plt.subplot(3, 2, 5)
            self._plot_monthly_summary_table(ax5)
            
            # 子图6: 统计摘要
            ax6 = plt.subplot(3, 2, 6)
            self._plot_summary_stats(ax6)
            
            # 保存第一页到PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 第二页：分类分析和支付方式
            fig2 = plt.figure(figsize=page_size)
            fig2.suptitle('分类分析与支付方式', fontsize=16, fontweight='bold', y=0.95)
            
            # 调整子图间距
            plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.1)
            
            # 支出品类分布饼图
            ax2_1 = plt.subplot(2, 2, 1)
            self._plot_expense_category_pie(ax2_1)
            
            # 收入来源分布饼图
            ax2_2 = plt.subplot(2, 2, 2)
            self._plot_income_source_pie(ax2_2)
            
            # 支付方式分析
            ax2_3 = plt.subplot(2, 1, 2)
            self._plot_payment_method_subplot(ax2_3)
            
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            # 第三页：详细数据表格（合并显示）
            fig3 = plt.figure(figsize=page_size)
            fig3.suptitle('详细交易记录', fontsize=16, fontweight='bold', y=0.95)
            
            # 创建合并的详细表格
            self._create_combined_detail_table(fig3)
            
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
        
        print(f"PDF文件已成功导出: {filename}")
        return filename
    
    def show_charts_with_options(self):
        """显示图表并提供PDF导出选项"""
        # 显示图表
        fig = self.show_all_charts()
        
        # 提供PDF导出选项
        try:
            print("\n" + "="*50)
            print("图表已显示完成！")
            print("是否需要导出PDF？")
            print("请输入文件名（不含扩展名），或按Enter跳过：")
            
            filename = input("PDF文件名: ").strip()
            
            if filename:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
                
                success = self.export_to_pdf(filename)
                if success:
                    print(f"✅ PDF已成功导出: {filename}")
                else:
                    print("❌ PDF导出失败")
            else:
                print("已跳过PDF导出")
                
        except KeyboardInterrupt:
            print("\n已取消PDF导出")
        except Exception as e:
            print(f"处理用户输入时发生错误: {e}")
        
        print("="*50)


def create_sample_data():
    """创建示例数据用于测试"""
    import random
    from datetime import datetime, timedelta
    
    # 示例交易分类
    income_categories = ['工资', '奖金', '投资收益', '兼职收入', '其他收入']
    expense_categories = ['餐饮', '交通', '购物', '娱乐', '房租', '水电费', '医疗', '教育']
    payment_methods = ['支付宝', '微信支付', '银行卡', '现金', '信用卡']
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(200):  # 生成200条示例数据
        # 随机生成日期
        random_days = random.randint(0, 365)
        transaction_date = start_date + timedelta(days=random_days)
        
        # 随机决定是收入还是支出
        is_income = random.choice([True, False])
        
        if is_income:
            category = random.choice(income_categories)
            amount = random.randint(3000, 15000)
            income_expense = '收入'
        else:
            category = random.choice(expense_categories)
            amount = -random.randint(50, 2000)
            income_expense = '支出'
        
        payment_method = random.choice(payment_methods)
        description = f"{category}相关交易"
        
        data.append({
            '交易时间': transaction_date,
            '交易分类': category,
            '商品说明': description,
            '收/支': income_expense,
            '金额': amount,
            '收/付款方式': payment_method
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # 创建示例数据进行测试
    print("创建示例数据进行图表测试...")
    sample_data = create_sample_data()
    
    # 创建图表可视化器
    visualizer = ChartVisualizer(sample_data)
    
    # 显示所有图表
    visualizer.show_all_charts()