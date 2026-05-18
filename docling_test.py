from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import PdfFormatOption, DocumentConverter, ImageFormatOption, PowerpointFormatOption, WordFormatOption, ExcelFormatOption, HTMLFormatOption

# 配置OCR模型，设置EasyOCR模型的路径
easyocr_model_storage_directory = "./models/docling/easyocr"
easyocr_options = EasyOcrOptions()

easyocr_options.lang = ['ch_sim','en']
easyocr_options.model_storage_directory = easyocr_model_storage_directory

# 配置pdf模型，设置Docling模型的路径
pdf_artifacts_path = "./models/docling/docling_model"

pipeline_options = PdfPipelineOptions(artifacts_path=pdf_artifacts_path)
# 设置支持OCR
pipeline_options.do_ocr = True
# 设置支持表结构
pipeline_options.do_table_structure = True

# 指定OCR模型
pipeline_options.ocr_options = easyocr_options

# 转换模型
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pipeline_options),
        InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options),
        InputFormat.XLSX: ExcelFormatOption(pipeline_options=pipeline_options),
        InputFormat.HTML: HTMLFormatOption(pipeline_options=pipeline_options)
    }
)

source = "./data/manual_pdf/convert_test.pdf"
output = "./data/manual_txt/convert_test.md"

result = converter.convert(source)
with open(output, "w", encoding = "utf-8") as f:
    f.write(result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED))

print("docling已完成解析！")



