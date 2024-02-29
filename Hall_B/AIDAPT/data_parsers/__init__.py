from Hall_B.AIDAPT.registration import make, register, list_registered_modules

register(
    id="aidapt_parser_v0", 
    entry_point= "Hall_B.AIDAPT.data_parsers.aidapt_parser_v0:AIDAPTParserV0"
)