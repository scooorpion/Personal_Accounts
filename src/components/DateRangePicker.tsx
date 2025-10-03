import React, { useState } from 'react';
import { Calendar } from '@/components/ui/calendar';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { CalendarIcon, X } from 'lucide-react';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { DateRange } from 'react-day-picker';
import { cn } from '@/lib/utils';
import { useFinancialStore } from '@/store/financialStore';

interface DateRangePickerProps {
  className?: string;
}

const DateRangePicker: React.FC<DateRangePickerProps> = ({ className }) => {
  const { dateRange, setDateRange } = useFinancialStore();
  const [isOpen, setIsOpen] = useState(false);
  
  const handleDateRangeChange = (range: DateRange | undefined) => {
    if (range) {
      setDateRange({
        startDate: range.from || null,
        endDate: range.to || null
      });
    }
  };
  
  const clearDateRange = () => {
    setDateRange({
      startDate: null,
      endDate: null
    });
    setIsOpen(false);
  };
  
  const formatDateRange = () => {
    if (!dateRange.startDate && !dateRange.endDate) {
      return '选择日期范围';
    }
    
    if (dateRange.startDate && dateRange.endDate) {
      return `${format(dateRange.startDate, 'yyyy年MM月dd日', { locale: zhCN })} - ${format(dateRange.endDate, 'yyyy年MM月dd日', { locale: zhCN })}`;
    }
    
    if (dateRange.startDate) {
      return `从 ${format(dateRange.startDate, 'yyyy年MM月dd日', { locale: zhCN })}`;
    }
    
    if (dateRange.endDate) {
      return `到 ${format(dateRange.endDate, 'yyyy年MM月dd日', { locale: zhCN })}`;
    }
    
    return '选择日期范围';
  };
  
  const hasDateRange = dateRange.startDate || dateRange.endDate;
  
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className={cn(
              'w-[280px] justify-start text-left font-normal',
              !hasDateRange && 'text-muted-foreground'
            )}
          >
            <CalendarIcon className="mr-2 h-4 w-4" />
            {formatDateRange()}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            initialFocus
            mode="range"
            defaultMonth={dateRange.startDate || new Date()}
            selected={{
              from: dateRange.startDate || undefined,
              to: dateRange.endDate || undefined
            }}
            onSelect={handleDateRangeChange}
            numberOfMonths={2}
            locale={zhCN}
          />
          <div className="p-3 border-t">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsOpen(false)}
              className="w-full"
            >
              确定
            </Button>
          </div>
        </PopoverContent>
      </Popover>
      
      {hasDateRange && (
        <Button
          variant="ghost"
          size="sm"
          onClick={clearDateRange}
          className="h-8 px-2 text-muted-foreground hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
};

export default DateRangePicker;