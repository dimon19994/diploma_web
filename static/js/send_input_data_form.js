$("#input_data").submit(function (e) {
    e.preventDefault();
    send_ajax("/import_data_from_file", "#input_data");
});

// TODO: Отображение точек на картинке
// $(document).on("click", "#points_type .col-10 + input[type=text]", function() {
//     alert("text");
// });

function send_ajax(url, data_tag) {
    $.ajax({
        url: url,
        type: "POST",
        data: new FormData($(data_tag)[0]),
        processData: false,
        contentType: false,
        success: function (response) {
            var data = JSON.parse(window.localStorage.getItem($("#file_name")[0].files[0].name));
            if (data == null)
                window.localStorage.setItem($("#file_name")[0].files[0].name, JSON.stringify({"data": response["data"]}));
            else {
                data["data"] = response["data"];
                $("#iter_count").val(data["iter_count"])
                $("#start_value").val(data["start_value"])
                $("#end_value").val(data["end_value"])
                $("#subitems").val(data["subitems"])
                $("#curve_type").val(data["curve_type"])
                if (data["curve_type"] == "not_loop") {
                    $("#align_1_container").attr("hidden", false);
                    $("#align_2_container").attr("hidden", false);
                    $("#align_1").val(data["align_1"])
                    $("#align_2").val(data["align_2"])
                } else {
                    $("#align_1_container").attr("hidden", true);
                    $("#align_2_container").attr("hidden", true);
                }
                // $("#direction_1").val(data["direction_1"])
                // $("#direction_2").val(data["direction_2"])
                window.localStorage.setItem($("#file_name")[0].files[0].name, JSON.stringify(data));
            }

            $("#points_type_pagination nav").remove()
            $("#points_type_pagination .pagination-page-info").remove()
            $("#save_input_data").remove()
            $("#points_type .col-3").empty()


            $("#input_data_img")[0].src = "data:image/png;base64," + response["plot"];
            $("#item-2").addClass("show");

            function template(data, pagination) {
                var data_local = JSON.parse(window.localStorage.getItem($("#file_name")[0].files[0].name))["data"];
                var html = '';
                var start = Number(pagination.pageSize * (pagination.pageNumber - 1));
                var column = 0
                for (let i = 0; i < data.length; i++) {
                    if (i % (pagination.pageSize / 4) == 0) {
                        column++;
                        html += '<div class="col-3 p-0 row" id="column-' + column + '">';
                    }
                    html += '<label class="col-2 my-1 col-form-label">' + (start + i + 1) + '.</label>\
                    <div class="col-10 my-1 row">\
                        <div class="p-0 col-3">\
                            <input type="text" class="form-control" id="pos-' + i + '-x" value="' + data[i][0] + '" disabled>\
                        </div>\
                        <div class="p-0 col-3">\
                            <input type="text" class="form-control" id="pos-' + i + '-y" value="' + data[i][1] + '" disabled>\
                        </div>\
                        <div class="p-0 col-6">\
                            <select id="pos-' + i + '-t" class="col-12 form-select" name="val-' + (start + i) + '" required>\
                                <option value="0" ' + (data_local[start + i][2] == 0 ? "selected" : "") + '>З пружинкою</option>\
                                <option value="1" ' + (data_local[start + i][2] == 1 ? "selected" : "") + '>Фіксовані</option>\
                                <option value="2" ' + (data_local[start + i][2] == 2 ? "selected" : "") + '>Уявні</option>\
                            </select>\
                        </div>\
                    </div>';
                    if (i % (pagination.pageSize / 4) == (pagination.pageSize / 4) - 1)
                        html += '</div>'
                }
                return html;
            }

            $('#pagination-container').pagination({
                dataSource: response['data'],
                pageSize: 40,
                callback: function (data, pagination) {
                    var html = template(data, pagination);
                    $('#data-container').html(html);
                }
            })

            $('<button type="button" class="m-3 btn btn-primary" id="save_input_data">Далі</button>').appendTo("#points_type_pagination");
        },
        error: function (jqXHR, textStatus, errorThrown) {
            // todo исправить ответ ошибки
            messenger.showErrorMessage(errorThrown);
        },
    });
};

$(document).on("click", "#save_input_data", function (e) {
    e.preventDefault();
    $("#item-2").removeClass("show");
    $("#item-3").addClass("show");
});

$(document).on("click", "#calculate", function (e) {
    e.preventDefault();
    $("#item-3").removeClass("show");
    $("#item-4").addClass("show");

    var local_data = JSON.parse(window.localStorage.getItem($("#file_name")[0].files[0].name));

    window.localStorage.setItem(
        $("#file_name")[0].files[0].name,
        JSON.stringify(
            {
                "data": local_data["data"],
                "iter_count": $("#iter_count").val(),
                "start_value": $("#start_value").val(),
                "end_value": $("#end_value").val(),
                "subitems": $("#subitems").val(),
                "curve_type": $("#curve_type").val(),
                "align_1": $("#align_1").val(),
                "align_2": $("#align_2").val(),
                // "direction_1": $("#direction_1").val(),
                // "direction_2": $("#direction_2").val(),
            }
        )
    );

    form_data = new FormData($("#build_params")[0])
    form_data.append("data", JSON.stringify(local_data["data"]))
    form_data.append("curve_type", $("#curve_type").val())
    // form_data.append("direction_1", $("#direction_1").val())
    // form_data.append("direction_2", $("#direction_2").val())
    form_data.append("save_data", $("#save_data")[0].checked ? 1 : 0)
    form_data.append("file_name", $("#file_name")[0].files[0].name)

    $.ajax({
        url: "/calculate",
        type: "POST",
        data: form_data,
        processData: false,
        contentType: false,
        success: function (response) {
            var max_iter = response["plots"].length;
            $("#image_box").empty();
            for (let i = 0; i < max_iter; i++) {
                $("#image_box").append(
                    '<li class="splide__slide"><img src="data:image/png;base64,' + response["plots"][i] + '"</li>'
                );
            }
            new Splide('#image-carousel', {
                wheel: true,
            }).mount();
            $("#quality").html("Якість = " + response["quality"]);
            $("#length").html("Довжина = " + response["length"]);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            // TODO: исправить ответ ошибки
            messenger.showErrorMessage(errorThrown);
        },
    });
});

$(document).on("change", "#data-container select", function (e) {
    var index = Number(e.currentTarget.name.split("-")[1])
    var value = Number(e.currentTarget.value)

    var data = JSON.parse(window.localStorage.getItem($("#file_name")[0].files[0].name));

    data["data"][index][2] = value;
    window.localStorage.setItem($("#file_name")[0].files[0].name, JSON.stringify(data));
})

$(document).on("change", "#range_iteration", function (e) {
    var value = e.val();
    $("#result_data_img")[0].src = "data:image/png;base64," + response["plots"][value];
    $("#text_iteration").val(max_iter + 1);
    $("#range_iteration").attr("max", max_iter).val(max_iter);
})

$(document).on("change", "#text_iteration", function (e) {
    $("#result_data_img")[0].src = "data:image/png;base64," + response["plots"][max_iter];
    $("#result_data_img").attr("hidden", false);
    $("#text_iteration").val(max_iter + 1);
    $("#range_iteration").attr("max", max_iter).val(max_iter);
})

$(document).on("change", "#curve_type", function (e) {
    if (e.currentTarget.value == "not_loop") {
        $("#align_1_container").attr("hidden", false);
        $("#align_2_container").attr("hidden", false);
    } else {
        $("#align_1_container").attr("hidden", true);
        $("#align_2_container").attr("hidden", true);
    }
});