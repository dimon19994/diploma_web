from flask import Blueprint, request

from controllers.main_page import MainPage
from controllers.import_data import ImportData
from controllers.calculate import Calculate


display = Blueprint("display", __name__)
# current_data = ""


@display.route("/", methods=["GET"])
def main():
    return MainPage(request).call()


@display.route("/import_data_from_file", methods=["POST"])
def import_data_from_file():
    return ImportData(request).call()


@display.route("/calculate", methods=["POST"])
def calculate():
    return Calculate(request).call()