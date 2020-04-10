if __name__ == '__main__':
    from pprint import pprint

    import execjs
    import pathlib
    import os

    js_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
    js_path = js_path / "encode.js"
    with js_path.open('r', encoding='utf-8') as f:
        script = f.read()

    script = script
    print("script", script)

    x = execjs.compile(script)
    result = x.call("encode","text=ku&page=1&type=migu")
    print(result)


