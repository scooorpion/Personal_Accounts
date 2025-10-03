import pandas as pd
import os
import glob
from chart_visualizer import ChartVisualizer

def find_header_row(file_path, keywords, max_rows_to_check=30):
    try:
        if file_path.endswith('.csv'):
            df_peek = pd.read_csv(file_path, header=None, nrows=max_rows_to_check, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df_peek = pd.read_excel(file_path, header=None, nrows=max_rows_to_check)
        else:
            print("不支持的文件格式。请提供.csv或.xlsx文件。")
            return None

        for i, row in df_peek.iterrows():
            row_str = ''.join(str(s) for s in row.values)
            if all(keyword in row_str for keyword in keywords):
                return i
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    return None

def process_wechat_file(file_path):
    """处理微信支付账单文件"""
    print(f"正在处理微信支付账单文件: {file_path}")
    
    try:
        # 微信支付账单从第17行开始（跳过前16行的头部信息）
        df = pd.read_excel(file_path, skiprows=16)
        
        # 微信支付账单的列名映射
        wechat_columns = {
            '交易时间': '交易时间',
            '交易类型': '交易分类', 
            '交易对方': '交易对方',
            '商品': '商品说明',
            '收/支': '收/支',
            '金额(元)': '金额',
            '支付方式': '收/付款方式'
        }
        
        # 重命名列以匹配统一格式
        df = df.rename(columns=wechat_columns)
        
        # 添加数据源标识
        df['数据源'] = '微信支付'
        
        # 过滤中性交易（收/支列为"/"的记录）
        if '收/支' in df.columns:
            # 查找中性交易（"/"符号）
            mask = df['收/支'].astype(str).str.strip() == '/'
            rows_to_delete = df[mask]
            
            if not rows_to_delete.empty:
                print(f"找到 {len(rows_to_delete)} 行中性交易数据，将被删除。")
                # 删除中性交易的行
                df = df[~mask].reset_index(drop=True)
                print(f"删除后剩余 {len(df)} 行数据。")
            else:
                print("未找到中性交易数据行。")
        
        # 数据清理
        if '交易时间' in df.columns:
            df['交易时间'] = pd.to_datetime(df['交易时间'], errors='coerce')
        
        if '金额' in df.columns:
            # 清理金额列，移除非数字字符
            df['金额'] = df['金额'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce')
        
        # 移除空行
        df = df.dropna(subset=['交易时间', '金额'], how='any')
        
        print(f"微信支付数据处理完成，共有 {len(df)} 行有效数据。")
        return df
        
    except Exception as e:
        print(f"处理微信支付文件时出错: {e}")
        return None

def process_alipay_file(file_path):
    """处理支付宝账单文件"""
    print(f"正在处理支付宝账单文件: {file_path}")
    
    header_keywords = ['交易时间', '交易分类', '商品说明']
    header_row_index = find_header_row(file_path, header_keywords)

    if header_row_index is None:
        print(f"错误：在文件 '{file_path}' 的前30行中未找到包含所有关键字的表头。")
        return None

    print(f"成功找到表头，位于第 {header_row_index + 1} 行。")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=header_row_index, encoding='utf-8')
        else: 
            df = pd.read_excel(file_path, header=header_row_index)
    except Exception as e:
        print(f"从第 {header_row_index + 1} 行读取数据时出错: {e}")
        return None
        
    # 添加数据源标识
    df['数据源'] = '支付宝'
    
    # 在收/支列搜索所有不计收支的列，然后删除这些行的数据
    if '收/支' in df.columns:
        # 查找包含"不计收支"的行
        mask = df['收/支'].astype(str).str.contains('不计收支', na=False)
        rows_to_delete = df[mask]
        
        if not rows_to_delete.empty:
            print(f"找到 {len(rows_to_delete)} 行包含'不计收支'的数据，将被删除。")
            # 删除包含"不计收支"的行
            df = df[~mask].reset_index(drop=True)
            print(f"删除后剩余 {len(df)} 行数据。")
        else:
            print("未找到包含'不计收支'的数据行。")

    # 数据清理和类型转换
    if '交易时间' in df.columns:
        df['交易时间'] = pd.to_datetime(df['交易时间'], errors='coerce')
    
    if '金额' in df.columns:
        # 清理金额列，移除非数字字符
        df['金额'] = df['金额'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce')
    
    # 移除空行
    df = df.dropna(subset=['交易时间', '金额'], how='any')
    
    print(f"支付宝数据处理完成，共有 {len(df)} 行有效数据。")
    return df

def process_transaction_file(file_path):
    """保留原有的单文件处理函数以保持兼容性"""
    if "微信支付账单" in file_path:
        return process_wechat_file(file_path)
    else:
        return process_alipay_file(file_path)

def find_and_process_all_files():
    """查找并处理所有支付账单文件"""
    current_dir = os.getcwd()
    all_dataframes = []
    
    # 查找微信支付账单文件
    wechat_files = glob.glob(os.path.join(current_dir, "*微信支付账单*.xlsx"))
    for file_path in wechat_files:
        print(f"找到微信支付账单文件: {file_path}")
        df = process_wechat_file(file_path)
        if df is not None and not df.empty:
            all_dataframes.append(df)
    
    # 查找支付宝账单文件
    alipay_files = glob.glob(os.path.join(current_dir, "*支付宝*.csv"))
    alipay_files.extend(glob.glob(os.path.join(current_dir, "*支付宝*.xlsx")))
    for file_path in alipay_files:
        print(f"找到支付宝账单文件: {file_path}")
        df = process_alipay_file(file_path)
        if df is not None and not df.empty:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("未找到任何支付账单文件")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 按时间排序
    if '交易时间' in combined_df.columns:
        combined_df = combined_df.sort_values('交易时间', ascending=False)
    
    print(f"数据合并完成，总共 {len(combined_df)} 行数据")
    print(f"其中微信支付: {len(combined_df[combined_df['数据源'] == '微信支付'])} 行")
    print(f"其中支付宝: {len(combined_df[combined_df['数据源'] == '支付宝'])} 行")
    
    return combined_df

if __name__ == "__main__":
    # 查找并处理所有支付账单文件
    combined_data = find_and_process_all_files()
    
    if combined_data is not None and not combined_data.empty:
        print("成功处理所有支付账单文件")
        print(f"合并后数据行数: {len(combined_data)}")
        
        # 创建图表可视化器并直接导出PDF
        visualizer = ChartVisualizer(combined_data)
        pdf_filename = visualizer.export_to_pdf()
        print(f"财务分析报告已生成: {pdf_filename}")
    else:
        print("未找到有效的支付账单文件，正在使用示例数据演示图表功能...")
        
        # 使用示例数据
        from chart_visualizer import create_sample_data
        sample_data = create_sample_data()
        print("示例数据已生成，开始创建图表...")
        
        # 创建图表可视化器并直接导出PDF
        visualizer = ChartVisualizer(sample_data)
        pdf_filename = visualizer.export_to_pdf()
        print(f"示例财务分析报告已生成: {pdf_filename}")