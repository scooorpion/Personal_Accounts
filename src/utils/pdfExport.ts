import jsPDF from 'jspdf';
import 'jspdf-autotable';
import html2canvas from 'html2canvas';
import { TransactionData } from '@/types/financial';
import { generateCategoryStats } from './dataProcessor';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

// 声明jsPDF的autoTable方法
declare module 'jspdf' {
  interface jsPDF {
    autoTable: (options: any) => jsPDF;
  }
}

export interface ExportData {
  totalIncome: number;
  totalExpense: number;
  transactionCount: number;
  dateRange: string;
  transactions: TransactionData[];
}

export const exportToPDF = async (chartsElementId: string, tablesElementId: string, data: ExportData) => {
  try {
    console.log('开始PDF导出，参数:', { chartsElementId, tablesElementId, dataLength: data.transactions.length });
    
    const chartsElement = document.getElementById(chartsElementId);
    const tablesElement = document.getElementById(tablesElementId);
    
    console.log('DOM元素检查:', { 
      chartsElement: !!chartsElement, 
      tablesElement: !!tablesElement,
      chartsElementSize: chartsElement ? { width: chartsElement.offsetWidth, height: chartsElement.offsetHeight } : null
    });
    
    if (!chartsElement || !tablesElement) {
      const error = `找不到要导出的元素: charts=${!!chartsElement}, tables=${!!tablesElement}`;
      console.error(error);
      throw new Error(error);
    }
    
    if (chartsElement.offsetWidth === 0 || chartsElement.offsetHeight === 0) {
      const error = '图表元素尺寸为0，可能未正确渲染';
      console.error(error);
      throw new Error(error);
    }

    console.log('图表元素尺寸:', {
      width: chartsElement.scrollWidth,
      height: chartsElement.scrollHeight,
      offsetWidth: chartsElement.offsetWidth,
      offsetHeight: chartsElement.offsetHeight
    });

    // 等待图表完全渲染
    await new Promise(resolve => {
      const checkCharts = () => {
        const svgs = chartsElement.querySelectorAll('svg');
        const canvases = chartsElement.querySelectorAll('canvas');
        const rechartContainers = chartsElement.querySelectorAll('.recharts-wrapper');
        
        console.log('图表渲染检查:', {
          svgCount: svgs.length,
          canvasCount: canvases.length,
          rechartContainers: rechartContainers.length
        });
        
        // 检查是否有足够的图表元素
        if (svgs.length > 0 || canvases.length > 0 || rechartContainers.length > 0) {
          resolve(true);
        } else {
          setTimeout(checkCharts, 100);
        }
      };
      
      setTimeout(checkCharts, 500); // 初始延迟
    });

    // 创建PDF文档，设置UTF-8编码支持
    const pdf = new jsPDF({
      orientation: 'p',
      unit: 'mm',
      format: 'a4',
      putOnlyUsedFonts: true,
      compress: true
    });
    
    // 设置字体以支持中文
    pdf.setFont('helvetica');
    
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 20;
    const contentWidth = pageWidth - 2 * margin;
    
    // 添加标题
    pdf.setFontSize(20);
    pdf.text('个人财务分析报告', pageWidth / 2, 20, { align: 'center' });
    
    // 添加生成时间
    pdf.setFontSize(12);
    const currentDate = new Date().toLocaleString('zh-CN');
    pdf.text(`生成时间: ${currentDate}`, margin, 35);
    
    // 添加数据摘要
    pdf.setFontSize(14);
    pdf.text('数据摘要', margin, 50);
    
    pdf.setFontSize(12);
    pdf.text(`分析期间: ${data.dateRange}`, margin, 65);
    pdf.text(`总收入: ¥${data.totalIncome.toLocaleString()}`, margin, 75);
    pdf.text(`总支出: ¥${data.totalExpense.toLocaleString()}`, margin, 85);
    pdf.text(`净收入: ¥${(data.totalIncome - data.totalExpense).toLocaleString()}`, margin, 95);
    pdf.text(`交易笔数: ${data.transactionCount}`, margin, 105);
    
    let currentY = 120;
    
    // 捕获图表区域
    console.log('开始捕获图表，html2canvas配置:', {
      scale: 1.5,
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff',
      width: chartsElement.scrollWidth,
      height: chartsElement.scrollHeight
    });
    
    const chartsCanvas = await html2canvas(chartsElement, {
      scale: 1.2,
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff',
      width: chartsElement.scrollWidth,
      height: chartsElement.scrollHeight,
      logging: false,
      foreignObjectRendering: true,
      removeContainer: true,
      imageTimeout: 15000,
      onclone: (clonedDoc, element) => {
        console.log('html2canvas克隆文档完成');
        // 确保所有SVG元素正确渲染
        const svgs = element.querySelectorAll('svg');
        svgs.forEach((svg) => {
          svg.style.overflow = 'visible';
          svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        });
        
        // 确保Canvas元素正确渲染
        const canvases = element.querySelectorAll('canvas');
        canvases.forEach((canvas) => {
          canvas.style.maxWidth = 'none';
          canvas.style.maxHeight = 'none';
        });
      }
    });
    
    console.log('图表捕获完成，canvas尺寸:', { width: chartsCanvas.width, height: chartsCanvas.height });
    
    const chartsImgData = chartsCanvas.toDataURL('image/png');
    const chartsImgWidth = contentWidth;
    const chartsImgHeight = (chartsCanvas.height * chartsImgWidth) / chartsCanvas.width;
    
    // 检查是否需要新页面放置图表
    if (currentY + chartsImgHeight > pageHeight - margin) {
      pdf.addPage();
      currentY = margin;
    }
    
    pdf.addImage(chartsImgData, 'PNG', margin, currentY, chartsImgWidth, chartsImgHeight);
    currentY += chartsImgHeight + 20;
    
    // 添加分类统计表格
    const incomeStats = generateCategoryStats(data.transactions, '收入');
    const expenseStats = generateCategoryStats(data.transactions, '支出');
    
    // 收入分类统计
    if (incomeStats.length > 0) {
      if (currentY + 60 > pageHeight - margin) {
        pdf.addPage();
        currentY = margin;
      }
      
      pdf.setFontSize(14);
      pdf.text('收入分类统计', margin, currentY);
      currentY += 15;
      
      // 使用autoTable创建表格，支持UTF-8编码
      const incomeTableData = incomeStats.slice(0, 10).map(stat => [
        stat.category,
        `¥${stat.amount.toLocaleString()}`,
        stat.count.toString(),
        `${stat.percentage.toFixed(1)}%`
      ]);
      
      pdf.autoTable({
        startY: currentY,
        head: [['分类', '金额', '笔数', '占比']],
        body: incomeTableData,
        styles: {
          font: 'helvetica',
          fontSize: 10,
          cellPadding: 3
        },
        headStyles: {
          fillColor: [66, 139, 202],
          textColor: 255,
          fontStyle: 'bold'
        },
        margin: { left: margin, right: margin },
        theme: 'striped'
      });
      
      currentY = (pdf as any).lastAutoTable.finalY + 10;
    }
    
    // 支出分类统计
    if (expenseStats.length > 0) {
      if (currentY + 60 > pageHeight - margin) {
        pdf.addPage();
        currentY = margin;
      }
      
      pdf.setFontSize(14);
      pdf.text('支出分类统计', margin, currentY);
      currentY += 15;
      
      // 使用autoTable创建表格，支持UTF-8编码
      const expenseTableData = expenseStats.slice(0, 10).map(stat => [
        stat.category,
        `¥${stat.amount.toLocaleString()}`,
        stat.count.toString(),
        `${stat.percentage.toFixed(1)}%`
      ]);
      
      pdf.autoTable({
        startY: currentY,
        head: [['分类', '金额', '笔数', '占比']],
        body: expenseTableData,
        styles: {
          font: 'helvetica',
          fontSize: 10,
          cellPadding: 3
        },
        headStyles: {
          fillColor: [220, 53, 69],
          textColor: 255,
          fontStyle: 'bold'
        },
        margin: { left: margin, right: margin },
        theme: 'striped'
      });
      
      currentY = (pdf as any).lastAutoTable.finalY + 20;
    }
    
    // 添加交易记录详情
    if (data.transactions.length > 0) {
      if (currentY + 60 > pageHeight - margin) {
        pdf.addPage();
        currentY = margin;
      }
      
      pdf.setFontSize(14);
      pdf.text('交易记录详情', margin, currentY);
      currentY += 15;
      
      // 准备交易记录数据（显示前50条）
      const transactionTableData = data.transactions.slice(0, 50).map(transaction => {
        const dateStr = format(transaction.transactionTime, 'MM-dd HH:mm', { locale: zhCN });
        const description = transaction.description.length > 20 ? 
          transaction.description.substring(0, 20) + '...' : transaction.description;
        const counterparty = transaction.counterparty.length > 15 ? 
          transaction.counterparty.substring(0, 15) + '...' : transaction.counterparty;
        
        return [
          dateStr,
          transaction.category,
          counterparty,
          description,
          transaction.type,
          `¥${transaction.amount.toLocaleString()}`
        ];
      });
      
      pdf.autoTable({
        startY: currentY,
        head: [['时间', '分类', '对方', '说明', '类型', '金额']],
        body: transactionTableData,
        styles: {
          font: 'helvetica',
          fontSize: 8,
          cellPadding: 2,
          overflow: 'linebreak',
          cellWidth: 'wrap'
        },
        headStyles: {
          fillColor: [108, 117, 125],
          textColor: 255,
          fontStyle: 'bold',
          fontSize: 9
        },
        columnStyles: {
          0: { cellWidth: 25 }, // 时间
          1: { cellWidth: 20 }, // 分类
          2: { cellWidth: 25 }, // 对方
          3: { cellWidth: 35 }, // 说明
          4: { cellWidth: 15 }, // 类型
          5: { cellWidth: 25 }  // 金额
        },
        margin: { left: margin, right: margin },
        theme: 'striped',
        pageBreak: 'auto'
      });
    }
    
    // 保存PDF
    const fileName = `财务分析报告_${new Date().toISOString().split('T')[0].replace(/-/g, '')}_${new Date().toTimeString().split(' ')[0].replace(/:/g, '')}.pdf`;
    console.log('准备保存PDF文件:', fileName);
    
    pdf.save(fileName);
    console.log('PDF导出成功完成');
    
    return true;
  } catch (error) {
    console.error('PDF导出失败，详细错误信息:', {
      message: error.message,
      stack: error.stack,
      name: error.name
    });
    
    // 提供更友好的错误信息
    let friendlyMessage = 'PDF导出失败，请重试';
    if (error instanceof Error) {
      if (error.message.includes('找不到要导出的元素') || error.message.includes('not found')) {
        friendlyMessage = '页面元素未找到，请确保数据已加载完成后再试';
      } else if (error.message.includes('尺寸为0') || error.message.includes('chart')) {
        friendlyMessage = '图表未完全加载，请等待图表渲染完成后重试';
      } else if (error.message.includes('html2canvas') || error.message.includes('canvas')) {
        friendlyMessage = '图表捕获失败，请尝试刷新页面后重新导出';
      } else if (error.message.includes('jsPDF') || error.message.includes('PDF')) {
        friendlyMessage = 'PDF生成失败，请检查数据完整性';
      } else if (error.message.includes('网络') || error.message.includes('network')) {
        friendlyMessage = '网络连接问题，请检查网络后重试';
      }
    }
    
    throw new Error(friendlyMessage);
  }
};

// 导出单个图表
export const exportChartToPDF = async (chartElementId: string, title: string) => {
  try {
    const element = document.getElementById(chartElementId);
    if (!element) {
      throw new Error('找不到要导出的图表元素');
    }

    // 创建PDF文档，设置UTF-8编码支持
    const pdf = new jsPDF({
      orientation: 'p',
      unit: 'mm',
      format: 'a4',
      putOnlyUsedFonts: true,
      compress: true
    });
    
    // 设置字体以支持中文
    pdf.setFont('helvetica');
    
    const pageWidth = pdf.internal.pageSize.getWidth();
    
    // 添加标题
    pdf.setFontSize(18);
    pdf.text(title, pageWidth / 2, 20, { align: 'center' });
    
    // 添加生成时间
    pdf.setFontSize(10);
    const currentDate = new Date().toLocaleString('zh-CN');
    pdf.text(`生成时间: ${currentDate}`, pageWidth / 2, 30, { align: 'center' });
    
    // 捕获图表
    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff'
    });
    
    const imgData = canvas.toDataURL('image/png');
    const imgWidth = pageWidth - 40;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    
    pdf.addImage(imgData, 'PNG', 20, 40, imgWidth, imgHeight);
    
    const fileName = `${title}_${new Date().toISOString().split('T')[0]}.pdf`;
    pdf.save(fileName);
    
    return true;
  } catch (error) {
    console.error('图表PDF导出失败:', error);
    throw error;
  }
};