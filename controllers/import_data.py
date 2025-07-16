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
        if "file_name" not in self.request.files:
            raise Exception("Invalid request, 'file' is required")

        file = self.request.files["file_name"]

        if file.filename == "":
            raise Exception("Invalid request, 'file' is required")

        separator = self.request.form["separator"]

        try:
            df = pd.read_csv(TextIOWrapper(file, encoding='utf-8-sig'), delimiter=separator, header=None)
        except csv.Error:
            raise Exception("Invalid file")

        M = len(df.values)
        input = df.values[::1].reshape(M, 2)
        input = np.insert(np.vstack([input, input[0]]), 2, 0, axis=1)


        plot = display_plot([np.vstack([input[:, 0], input[:, 1]])], labels=['Initial input'], color_line=['-ob'], title="",
                            annotate_step=[5], points_count=M)

        # ylim = plot.gca().get_ylim()
        # xlim = plot.gca().get_xlim()
        flike = BytesIO()
        plot.savefig(flike, format='png')
        flike.seek(0)
        image_png = flike.getvalue()
        graph = base64.b64encode(image_png).decode('utf-8')
        flike.close()

        return {"plot": graph, "data": input[:-1].tolist()}

    def _get(self):
        return render_template("main_page.html")
