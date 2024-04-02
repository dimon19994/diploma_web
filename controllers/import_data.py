import csv
from io import TextIOWrapper, BytesIO
import base64

import numpy as np
import pandas as pd
from flask import render_template

from controllers import _Controller
from utils import display_plot


class ImportData(_Controller):
    def _post(self):
        self.get_request_data()
        self.verify_required_fields(("separator",))

        separator = self.request.form["separator"]
        df = self._verify_and_get_file(separator)

        size = len(df.values)
        raw_input = df.values.reshape(size, 2)
        input_data = np.insert(raw_input, 2, 0, axis=1)

        plot = display_plot(
            [np.vstack([input_data[:, 0], input_data[:, 1]])],
            labels=["Initial input"],
            color_line=["-ob"],
            title="",
            annotate_step=[5],
            points_count=size,
        )

        flike = BytesIO()
        plot.savefig(flike, format="png")
        flike.seek(0)
        image_png = flike.getvalue()
        graph = base64.b64encode(image_png).decode("utf-8")
        flike.close()

        return {"plot": graph, "data": input_data.tolist()}

    def _get(self):
        return render_template("main_page.html")

    def _verify_and_get_file(self, separator):
        if "file_name" not in self.request.files:
            raise Exception("Invalid request, 'file' is required")

        file = self.request.files["file_name"]

        if file.filename == "":
            raise Exception("Invalid request, 'file' is required")

        try:
            df = pd.read_csv(
                TextIOWrapper(file, encoding="utf-8-sig"),
                delimiter=separator,
                header=None,
            )
        except csv.Error:
            raise Exception("Invalid file")

        return df
