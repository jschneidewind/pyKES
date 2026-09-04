Streamlit application (``pyKES.streamlit_app``)
===============================================

Reusable pages an external repository configures rather than forks. See
:doc:`/guide/streamlit_app`.

Configuration interface
-----------------------

The extension point: an embedding repository customizes the pages by
constructing these dataclasses.

.. automodule:: pyKES.streamlit_app.config_interface
   :members:

Chunked processing
------------------

Advancing a long processing run one experiment per Streamlit rerun, so the
progress bar stays alive in the browser. See :doc:`/browser_deployment`.

.. automodule:: pyKES.streamlit_app.chunked_processing
   :members:

Page components
---------------

Each is a page entry point taking, at most, its configuration dataclass.

.. automodule:: pyKES.streamlit_app.components.home_component
   :members:

.. automodule:: pyKES.streamlit_app.components.data_upload_component
   :members:

.. automodule:: pyKES.streamlit_app.components.analysis_results_component
   :members:

.. automodule:: pyKES.streamlit_app.components.time_series_component
   :members:

.. automodule:: pyKES.streamlit_app.components.results_table_component
   :members:
