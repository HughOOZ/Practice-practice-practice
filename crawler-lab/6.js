function getparam() {
    "use strict";
    let base64 = new Base64();
    let key = uuid();
    let time = (Math.floor((new Date().getTime() + 10010) / 99)).toString();
    console.log(key, time);
    let sign = md5(key + base64.encode(time) + 'xianyucoder11');
    let param = {"key": key, "time": time, "sign": sign};
    return param
};render(getparam())