from aidapt_toolkit.registration import make, register, list_registered_modules

register(
    id="lab_variables_to_invariants", 
    entry_point= "aidapt_toolkit.data_prep.lab_variables_to_invariants:LabVariablesToInvariants"
)

register(
    id="numpy_standard_scaler", 
    entry_point= "aidapt_toolkit.data_prep.numpy_standard_scaler:NumpyStandardScaler"
)

register(
    id="numpy_minmax_scaler", 
    entry_point= "aidapt_toolkit.data_prep.numpy_minmax_scaler:NumpyMinMaxScaler"
)