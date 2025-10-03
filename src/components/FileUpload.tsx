import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle2, Plus, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useFinancialStore } from '@/store/financialStore';

const FileUpload: React.FC = () => {
  const { uploadFiles, uploadStatus, clearData, transactions } = useFinancialStore();
  const [dragActive, setDragActive] = useState(false);
  const [uploadMode, setUploadMode] = useState<'cumulative' | 'replace'>('cumulative');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      uploadFiles(acceptedFiles, uploadMode === 'cumulative');
    }
  }, [uploadFiles, uploadMode]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv']
    },
    multiple: true,
    disabled: uploadStatus.isUploading
  });

  const handleClearData = () => {
    clearData();
  };

  return (
    <div className="space-y-4">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>上传账单文件</span>
            {transactions.length > 0 && (
              <span className="text-sm font-normal text-gray-600">
                当前已有 {transactions.length} 条交易记录
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
              ${uploadStatus.isUploading ? 'cursor-not-allowed opacity-50' : ''}
            `}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 bg-gray-100 rounded-full">
                <Upload className="h-8 w-8 text-gray-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-gray-900">
                  {isDragActive ? '释放文件到这里' : 
                   transactions.length > 0 ? '继续添加账单文件' : '上传账单文件'}
                </h3>
                <p className="text-sm text-gray-600">
                  支持微信支付和支付宝账单文件（.xlsx, .xls, .csv）
                </p>
                <p className="text-xs text-gray-500">
                  {transactions.length > 0 
                    ? '新上传的文件将与现有数据合并，重复交易会自动去重'
                    : '拖拽文件到这里或点击选择文件'}
                </p>
              </div>
              {!uploadStatus.isUploading && (
                <div className="flex flex-col items-center space-y-3 mt-4">
                  <Button variant="outline">
                    <FileText className="h-4 w-4 mr-2" />
                    选择文件
                  </Button>
                  
                  {transactions.length > 0 && (
                    <div className="flex items-center space-x-2 text-xs">
                      <Button
                        variant={uploadMode === 'cumulative' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setUploadMode('cumulative')}
                      >
                        <Plus className="h-3 w-3 mr-1" />
                        累加模式
                      </Button>
                      <Button
                        variant={uploadMode === 'replace' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setUploadMode('replace')}
                      >
                        <RotateCcw className="h-3 w-3 mr-1" />
                        替换模式
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 上传进度 */}
      {uploadStatus.isUploading && (
        <Card>
          <CardContent className="p-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">正在处理文件...</span>
                <span className="text-sm text-gray-500">{Math.round(uploadStatus.progress)}%</span>
              </div>
              <Progress value={uploadStatus.progress} className="w-full" />
            </div>
          </CardContent>
        </Card>
      )}

      {/* 错误信息 */}
      {uploadStatus.error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {uploadStatus.error}
          </AlertDescription>
        </Alert>
      )}

      {/* 成功信息 */}
      {uploadStatus.success && (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            文件上传成功！数据已准备就绪。
          </AlertDescription>
        </Alert>
      )}

      {/* 清空数据按钮 */}
      {uploadStatus.success && (
        <div className="flex justify-end">
          <Button variant="outline" onClick={handleClearData}>
            清空数据
          </Button>
        </div>
      )}
    </div>
  );
};

export default FileUpload;