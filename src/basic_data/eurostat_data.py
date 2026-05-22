import eurostat
import matplotlib.pyplot as plt
import pandas as pd

RAW_DATASET_CODES = ['A-T_Z', 'A-T_Z_XNB', 'A00-A08', 'A09', 'A15-A19_B90', 'A40_A41', 'ABORT_OTH', 'ARTHROPAT_OTH', 'A_B', 'A_B_OTH', 'B20-B24', 'C00-D48', 'C18-C21', 'C33_C34', 'C43_C44', 'C50', 'C53-C55', 'C56', 'C61', 'C67', 'C_OTH', 'D00-D09', 'D00-D48_OTH', 'D12', 'D25', 'D50-D64', 'D50-D89', 'D65-D89', 'E', 'E10-E14', 'E_OTH', 'F', 'F00-F03', 'F10', 'F11-F19', 'F20-F29', 'F30-F39', 'F_OTH', 'G', 'G30', 'G35', 'G40_G41', 'G45', 'G_OTH', 'H00-H59', 'H00-H59_OTH', 'H25_H26_H28', 'H60-H95', 'I', 'I10-I15', 'I20', 'I21_I22', 'I23-I25', 'I26-I28', 'I44-I49', 'I50', 'I60-I69', 'I70', 'I83', 'INJ_HEAD_OTH', 'INJ_OTH', 'INTESTINE_OTH', 'I_OTH', 'J', 'J00-J11', 'J12-J18', 'J20-J22', 'J35', 'J40-J44_J47', 'J45_J46', 'J60-J99', 'K', 'K00-K08', 'K09-K14', 'K20-K23', 'K25-K28', 'K29-K31', 'K35-K38', 'K40', 'K41-K46', 'K50_K51', 'K52', 'K56', 'K57', 'K60-K62', 'K70', 'K71-K77', 'K80', 'K81-K83', 'K85-K87', 'K_OTH', 'L', 'L00-L08', 'L20-L45', 'L_OTH', 'M', 'M16', 'M17', 'M23', 'M30-M36', 'M40-M49', 'M50_M51', 'M53_M80-M99', 'M54', 'M60-M79', 'N', 'N00-N16', 'N17-N19', 'N20-N23', 'N25-N39', 'N40', 'N41-N51', 'N60-N64', 'N70-N77', 'N91-N95', 'N_OTH', 'O', 'O04', 'O10-O48', 'O60-O75', 'O80', 'O81-O84', 'O85-O92', 'O95-O99', 'P', 'P07', 'P_OTH', 'Q', 'R', 'R07', 'R10', 'R69', 'R_OTH', 'S06', 'S52', 'S72', 'S82', 'S_T', 'S_T_OTH', 'T20-T32', 'T36-T65', 'T80-T88', 'T90-T98', 'UPRESPIR_OTH', 'U_COV19', 'U_COV19_OTH', 'Z', 'Z03', 'Z30', 'Z38', 'Z51', 'Z_OTH']

INPUT_ICD_CODE = 'A09.9'
TOP_AGES = 15
MAKE_PLOT = True



def get_dataset_code(input: str) -> str | None:
    if "." in input:
        input = input.split(".")[0]
    if input in RAW_DATASET_CODES:
        return input
    else:
        selection = [code for code in RAW_DATASET_CODES if input[0] in code]
        if len(selection) == 1:
            return selection[0]
        else:
            for c in selection:
                if "_" in c:
                    c_splitted = c.split("_")
                    if input in c_splitted:
                        return c
                    else:
                        for cs in c_splitted:
                            if "-" in c_splitted:
                                c_inter = cs.split("-")
                                a = int(c_inter[0][1:])
                                b = int(c_inter[1][1:])
                                if (a <= int(input[1:]) <= b) and input[0] == c_inter[0][0]:
                                    return c
                        if input[0] in c_splitted:
                             return c
    if input[0] in RAW_DATASET_CODES:
        return input[0]
    return None

def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    geo_col = "geo\\TIME_PERIOD"
    year_cols = [col for col in df.columns if str(col).isdigit()]
    long_df = df.rename(columns={geo_col: "geo"}).melt(
        id_vars=[c for c in df.columns if c not in year_cols and c != geo_col]
        + ["geo"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long_df["year"] = long_df["year"].astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.dropna(subset=["value"])






def save_plot(age_prob: pd.DataFrame, sex_prob: pd.DataFrame, output_path: str, dataset_code: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    age_plot = age_prob.sort_values("probability")
    axes[0].barh(age_plot["age"], age_plot["probability"], color="#4e79a7")
    axes[0].set_title(f"P(age | ICD-10: {dataset_code})")
    axes[0].set_xlabel("Probability")

    axes[1].bar(sex_prob["sex"], sex_prob["probability"], color="#f28e2b")
    axes[1].set_title(f"P(sex | ICD-10: {dataset_code})")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main() -> None:
    selected_icd = INPUT_ICD_CODE
    output_graph = f"icd10_{selected_icd.replace('.', '_')}_probabilities.png"

    df = eurostat.get_data_df("hlth_co_disch2")
    long_df = to_long_format(df)

    dataset_code = get_dataset_code(INPUT_ICD_CODE)
    print(f"Dataset code: {dataset_code}")
    if dataset_code is None:
        print(f"Dataset code not found for ICD-10: {INPUT_ICD_CODE}")
        return
    filtered = long_df[
        (long_df["freq"] == "A")
        & (long_df["indic_he"] == "INPAT")
        & (long_df["unit"] == "P_HTHAB")
        & (long_df["sex"].isin(["M", "F"]))
        & (long_df["age"] != "TOTAL")
        & (long_df["icd10"] == dataset_code)
    ].copy()

    if filtered.empty:
        print("No records for selected filters (country/year).")
        return


    selected = filtered.copy()
    if selected.empty:
        print("No records for selected ICD-10 code after filtering.")
        return

    total = selected["value"].sum()
    if total <= 0:
        print("Total value is zero; probabilities cannot be computed.")
        return

    age_prob = (
        selected.groupby("age", as_index=False)["value"]
        .sum()
        .assign(probability=lambda d: d["value"] / total)
        .sort_values("probability", ascending=False)
    )

    sex_prob = (
        selected.groupby("sex", as_index=False)["value"]
        .sum()
        .assign(probability=lambda d: d["value"] / total)
        .sort_values("probability", ascending=False)
    )

    print(f"ICD-10: {selected_icd} | Countries: ALL | Years: ALL")
    print()
    print(f"Top {TOP_AGES} probabilities by age group (P(age | ICD-10)):")
    print(age_prob.head(TOP_AGES).assign(probability_pct=lambda d: (d["probability"] * 100).round(2))[["age", "value", "probability", "probability_pct"]].to_string(index=False))
    print()
    print("Probabilities by sex (P(sex | ICD-10)):")
    print(
        sex_prob.assign(
            probability_pct=lambda d: (d["probability"] * 100).round(2)
        )[["sex", "value", "probability", "probability_pct"]].to_string(index=False)
    )

    if MAKE_PLOT:
        save_plot(age_prob.head(TOP_AGES), sex_prob, output_graph, dataset_code)
        print()
        print(f"Graph saved to: {output_graph}")


if __name__ == "__main__":
    main()