from flask import render_template

from controllers import _Controller


class MainPage(_Controller):
    def _post(self):
        pass

    def _get(self):
        return render_template("main_page.html")
