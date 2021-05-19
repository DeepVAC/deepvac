from .log import LOG

def addUserConfig(module_name, config_name, user_give=None, developer_give=None, is_user_mandatory=False):
    if user_give is None and is_user_mandatory:
        LOG.logE("You must set mandatory {}.{} in config.py. Developer advised value:{}".format(module_name, config_name, developer_give),exit=True)
    if user_give is not None:
        return user_give
    if developer_give is not None:
        return developer_give
    LOG.logE("value missing for configuration: {}.{} in config.py".format(module_name, config_name), exit=True)