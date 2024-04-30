import streamlit as st
from abas_ml_lib import Read_And_EDA, AutoML  # Replace 'your_module' with the name of your module
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title("AutoML Web Application")

    # Ask user to enter the path of their data file
    file_path = st.text_input("Enter the path of your data file")

    # Ask user to enter the target column
    target_column = st.text_input("Enter the target column name")

    best_model_info = None  # Initialize best_model_info here

    # Perform EDA and ML
    if st.button("Process Data"):
        if file_path.strip() == "":
            st.error("Please enter the path of your data file")
        elif target_column.strip() == "":
            st.error("Please enter the target column name")
        else:
            st.write("Reading the data file...")
            read_eda = Read_And_EDA()
            df = read_eda.readFile(file_path)

            if df is not None:
                st.write("Performing Exploratory Data Analysis...")
                clean_data = read_eda.do_EDA(df)
                st.write("Exploratory Data Analysis Complete")

                st.write("Finding the best model...")
                automl = AutoML(clean_data, target_column)
                best_model_info = automl.find_best_model()
                st.write("Best Model:", best_model_info)  # Display best_model_info
                st.write("Model training and evaluation complete")

                # Display correlation heatmap
                st.write("Correlation Heatmap:")
                plt.figure(figsize=(10, 8))
                sns.heatmap(clean_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
                st.pyplot()

                # Display cleaned data
                st.write("Cleaned Data:")
                st.dataframe(clean_data)
            else:
                st.error("Failed to read the data file. Please check the file path.")

    # Display best_model_info here if available
    if best_model_info is not None:
        st.write("Best Model Information (outside button click):", best_model_info)

if __name__ == "__main__":
    main()
