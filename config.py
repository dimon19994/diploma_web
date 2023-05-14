# from environs import Env
#
# env = Env()
#
# env.read_env()
#
# with env.prefixed("DIPLOMA_WEB_"):
#
#     with env.prefixed("DB_"):
#         DB_CONFIG = dict(
#             database=env.str("NAME"),
#             user=env.str("USER"),
#             password=env.str("PASSWORD"),
#             host=env.str("HOST"),
#             port=env.int("PORT"),
#             autorollback=True,
#         )
#
#     class BaseConfig(object):
#         PROJECT = "diploma-py"
#         DEBUG = False
#         TESTING = False
#         SECRET_KEY = env.str("SECRET_KEY")
#
#
#     class TestConfig(BaseConfig):
#         DEBUG = True
#         TESTING = True
#         PRESERVE_CONTEXT_ON_EXCEPTION = False
#         WTF_CSRF_ENABLED = False
