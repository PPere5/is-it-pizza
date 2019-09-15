def give_infos(bool_pizza):

    if bool_pizza:
        disp_text = "Yes it is Pizza! Yum!"
    else:
        disp_text = "No, that is not Pizza!"

    infos = [
        {
            'disp_text': disp_text
        }
    ]

    return infos



