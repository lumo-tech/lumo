"""

"""

from lumo.utils.markdown import Markdown

if __name__ == '__main__':
    md = Markdown()
    md.add_title("report of exp")
    md.add_title("All loss report")
    md.add_table([[1, 2, 3], [4, 5, 6]])
    md.add_text("Today alisjoialn")
    md.add_url("www.baidu.com", "baidu")
    md.add_bold("aoisjdoij")
    md.add_picture("ww.bia", "ada", True)

    md.add_code("""
    print(asdas)
    for i in range():
        print(1)
    """)

    print(md)
