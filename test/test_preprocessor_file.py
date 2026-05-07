import pandas as pd
from Core.preprocessor import EDA

data = pd.read_csv("diabetes.csv")
print(EDA.detect_object_type(data))
# option = st.selectbox(
#                 "Do you want to delete rows or columns?",
#                 ("delete row", "delete col")
#             )
#             if st.button("delete missing value"):
#                 if option == "delete row":
#                     df = MissingValue.remove_missing_values(data=df, axis=0)
#                     st.session_state['df'] = df
#                     st.session_state['success_msg'] = "✅ All missing values were deleted!"
#                     st.success(f"✅ All missing values were deleted!")
#                     st.rerun()
#                 elif option == "delete col":
#                     st.warning("Deleting in column mode will delete the columns")
#                     df = MissingValue.remove_missing_values(data=df, axis=1)
#                     st.session_state['df'] = df
#                     st.session_state['success_msg'] = "✅ All missing values were deleted!"
#                     st.success("✅ All missing values were deleted!")
#                     st.rerun()