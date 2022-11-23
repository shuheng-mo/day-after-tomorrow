#######
Andrew
#######

Andrew's Module and functions
------------------------------

This package implements LSTM and Convolutional LSTM with PCA

See :`tools` folder for more information.


dataprocessing.py
------------------------------
.. automodule:: tools
  :members: dataprocessing

.. automodule:: tools.dataprocessing
  :members: Load_data_to_train, Load_data_to_test, manipdata, data_storm_id, Loader_to_1D_array, SplitData, create_inout_seq
  :noindex: dataprocessing


fc_lstm.py
------------------------------
.. automodule:: tools
  :members: fc_lstm

.. automodule:: tools.fc_lstm
  :members: set_seed, train_lstm_mse, validate_mse, train_lstm_SSIM, validate_SSIM, evaluate, train_mse, train_ssim
  :noindex: fc_lstm


visualisation.py
------------------------------
.. automodule:: tools
  :members: visualisation

.. automodule:: tools.visualisation
  :members: show_batch
  :noindex: visualisation


metric.py
------------------------------
.. automodule:: tools
  :members: metric

.. automodule:: tools.metric
  :members: Generate_ssim_mse
  :noindex: metric


fc_lstm_pca.py
------------------------------
.. automodule:: tools
  :members: fc_lstm_pca

.. automodule:: tools.fc_lstm_pca
  :members: evaluate_pca
  :noindex: fc_lstm_pca
  
.. rubric:: References

