class _Controller(object):
    def __init__(self, request):
        self.request = request
        self.request_data = self.get_request_data()

    def call(self, *args, **kwargs):
        data = "Error"
        try:
            if self.request.method == "GET":
                data = self._get(*args, **kwargs)
            else:
                data = self._post(*args, **kwargs)
        except Exception as ex:
            print(str(ex))
            print("Error during %s call" % self.__class__.__name__)
        finally:
            return data

    def _get(self, *args, **kwargs):
        raise NotImplementedError("_get from %s" % self.__class__.__name__)

    def _post(self, *args, **kwargs):
        raise NotImplementedError("_post from %s" % self.__class__.__name__)

    def get_request_data(self):
        return dict(
            self.request.json
            if self.request.is_json
            else (self.request.form.items() or {})
        )

    def verify_required_fields(self, required_fields):
        for field in required_fields:
            if field not in self.request_data:
                raise Exception(f"Invalid request, '{field}' is required")
