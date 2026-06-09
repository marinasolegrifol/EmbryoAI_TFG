import pandas as pd

FILES = {
    "ResNet":       r"C:\Users\oriol\Desktop\test_results\ResNet\predictions_rn.xlsx",
    "MobileNet":    r"C:\Users\oriol\Desktop\test_results\MobileNet\predictions_mn.xlsx",
    "EfficientNet": r"C:\Users\oriol\Desktop\test_results\EfficientNet\predictions_en.xlsx",
}

OUTPUT = r"C:\Users\oriol\Desktop\test_results\prediccions.xlsx"

KEEP_ROW_LABELS = False

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    for sheet, path in FILES.items():
        if KEEP_ROW_LABELS:
            df = pd.read_excel(path, index_col=0)
            df.to_excel(writer, sheet_name=sheet[:31], index=True)
        else:
            df = pd.read_excel(path)
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

print("FET -> ", OUTPUT)