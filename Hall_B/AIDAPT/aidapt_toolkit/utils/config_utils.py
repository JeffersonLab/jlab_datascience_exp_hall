def verify_config(config, required_keys):
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        error_msg = f'Config is missing the following required keys: {", ".join(missing_keys)}'
        raise KeyError(error_msg)