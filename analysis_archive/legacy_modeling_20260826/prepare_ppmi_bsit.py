"""Build the reproducible B-SIT 12 + SCOPA-AUT training table from PPMI exports."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPSIT_FILE = DATA_DIR / "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_12Apr2026.csv"
SCOPA_FILE = DATA_DIR / "SCOPA-AUT_12Apr2026.csv"
STATUS_FILE = DATA_DIR / "Participant_Status_12Apr2026.csv"
CODE_LIST_FILE = DATA_DIR / "codeList" / "Code_List_-__Annotated__13Apr2026.csv"
OUTPUT_FILE = DATA_DIR / "PPMI_BSIT12_SCOPA.csv"

# The item numbers are verified against the supplied PPMI code list and the
# response code observed where SCENT_nn_CORRECT == 1. Do not replace these with
# the first 12 UPSIT items: B-SIT is a specific 12-odor subset.
BSIT_SCENT_MAP = {
    "BSIT_CHERRY": (4, "Cherry"),
    "BSIT_DILL_PICKLE": (25, "Dill Pickle"),
    "BSIT_BANANA": (7, "Banana"),
    "BSIT_CHOCOLATE": (19, "Chocolate"),
    "BSIT_CINNAMON": (15, "Cinnamon"),
    "BSIT_GASOLINE": (16, "Gasoline"),
    "BSIT_LEMON": (36, "Lemon"),
    "BSIT_ONION": (11, "Onion"),
    "BSIT_PINEAPPLE": (26, "Pineapple"),
    "BSIT_ROSE": (39, "Rose"),
    "BSIT_SOAP": (37, "Soap"),
    "BSIT_SMOKE": (33, "Smoke"),
}


def verify_bsit_mapping(upsit: pd.DataFrame, code_list: pd.DataFrame) -> None:
    """Fail closed if the supplied export/code list no longer supports the mapping."""
    for feature, (number, expected_odor) in BSIT_SCENT_MAP.items():
        correct_col = f"SCENT_{number:02d}_CORRECT"
        response_col = f"SCENT_{number:02d}_RESPONSE"
        correct_codes = (
            pd.to_numeric(upsit.loc[upsit[correct_col].eq(1), response_col], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(correct_codes) != 1:
            raise ValueError(f"{feature}: expected one correct response code, got {correct_codes}")
        decoded = code_list.loc[
            code_list["MOD_NAME"].eq("UPSIT")
            & code_list["ITM_NAME"].eq(response_col)
            & pd.to_numeric(code_list["CODE"], errors="coerce").eq(correct_codes[0]),
            "DECODE",
        ].astype(str).unique()
        if len(decoded) != 1 or decoded[0].casefold() != expected_odor.casefold():
            raise ValueError(f"{feature}: mapping mismatch, expected {expected_odor}, got {decoded}")


def build_training_table() -> pd.DataFrame:
    upsit = pd.read_csv(UPSIT_FILE, low_memory=False)
    scopa = pd.read_csv(SCOPA_FILE, low_memory=False)
    status = pd.read_csv(STATUS_FILE, low_memory=False)
    code_list = pd.read_csv(CODE_LIST_FILE, low_memory=False)
    verify_bsit_mapping(upsit, code_list)

    # Use the same baseline visit for both assessments. Mixing visits would leak
    # longitudinal information and make the kiosk's one-time screen hard to interpret.
    upsit = upsit.loc[upsit["EVENT_ID"].eq("BL")].copy()
    scopa = scopa.loc[scopa["EVENT_ID"].eq("BL")].copy()
    status = status.loc[status["COHORT"].isin([1, 2]), ["PATNO", "COHORT"]].copy()

    selected = upsit[["PATNO", "EVENT_ID"]].copy()
    for feature, (number, _) in BSIT_SCENT_MAP.items():
        selected[feature] = pd.to_numeric(
            upsit[f"SCENT_{number:02d}_CORRECT"], errors="coerce"
        )

    scopa = scopa[["PATNO", "EVENT_ID", "SCAU5", "SCAU6", "SCAU7"]].rename(
        columns={"SCAU5": "SCOPA_AUT5", "SCAU6": "SCOPA_AUT6", "SCAU7": "SCOPA_AUT7"}
    )
    merged = selected.merge(scopa, on=["PATNO", "EVENT_ID"], validate="one_to_one")
    merged = merged.merge(status, on="PATNO", validate="many_to_one")
    merged["LABEL_PD"] = merged["COHORT"].eq(1).astype(int)

    feature_cols = list(BSIT_SCENT_MAP) + ["SCOPA_AUT5", "SCOPA_AUT6", "SCOPA_AUT7"]
    merged = merged.dropna(subset=feature_cols).copy()
    merged[feature_cols] = merged[feature_cols].astype(float)
    output_cols = ["PATNO", "EVENT_ID", *feature_cols, "LABEL_PD"]
    result = merged[output_cols].sort_values("PATNO").reset_index(drop=True)
    result.to_csv(OUTPUT_FILE, index=False)
    return result


if __name__ == "__main__":
    table = build_training_table()
    print(
        f"saved={OUTPUT_FILE} rows={len(table)} "
        f"PD={int(table['LABEL_PD'].sum())} HC={int((table['LABEL_PD'] == 0).sum())}"
    )
