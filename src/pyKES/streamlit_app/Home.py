import streamlit as st
import os
import tempfile

from pyKES.database.database_experiments import ExperimentalDataset

st.set_page_config(
        page_title="Kinetic Data Visualization", 
        page_icon=":microscope:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Title
st.title("Kinetic Data Visualization")

st.sidebar.success("Select a page above.")

# Initialize session state for shared data
if 'experimental_dataset' not in st.session_state:
    st.session_state.experimental_dataset = None

uploaded_file = st.file_uploader("Upload HDF5 File", type=['h5', 'hdf5'])

if uploaded_file is not None:
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            # Write the contents of the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load the spectral dataset from the temporary file
            st.session_state.experimental_dataset = ExperimentalDataset.load_from_hdf5(tmp_file_path)
            st.success("File uploaded successfully! You can now navigate to other pages to analyze this data.")

        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    except Exception as e:
        st.error(f"Error loading HDF5 file: {str(e)}")
        st.info("Please ensure the HDF5 file has the correct structure with experimental data and metadata.")
else:
    st.info("Please upload an HDF5 file to visualize experimental data.")


if st.session_state.experimental_dataset is not None:
    st.title("Loaded Dataset")
    st.dataframe(st.session_state.experimental_dataset.overview_df, use_container_width=True)

st.markdown("""
This is the home page of our data analysis application.
Choose a page from the sidebar to begin your analysis:
* **Data Upload and Download** - Adding new data and downloading dataset
* **📈 Data Analysis** - Basic statistical analysis
* **📊 Visualization** - Data visualization tools
""")

