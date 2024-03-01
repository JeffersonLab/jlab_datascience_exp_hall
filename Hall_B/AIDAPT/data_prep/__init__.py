from Hall_B.AIDAPT.registration import make, register, list_registered_modules

register(
    id="lab_variables_to_invariants", 
    entry_point= "Hall_B.AIDAPT.data_parsers.lab_variables_to_invariants:LabVariablesToInvariants"
)