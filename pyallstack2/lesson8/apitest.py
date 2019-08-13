import unittest
import json
import requests

class ApiDemoTest(unittest.TestCase):
    @classmethod
    def get_url(self, url):
        r=requests.get(url)
        return (r.status_code, json.loads(r.text))

    def test_url1(self):
        url="https://www.apiopen.top/journalismApi"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url2(self):
        url="https://www.apiopen.top/satinGodApi?type=1&page=1"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url3(self):
        url="https://www.apiopen.top/novelApi"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url4(self):
        url="https://www.apiopen.top/novelSearchApi?name=%E7%9B%97%E5%A2%93%E7%AC%94%E8%AE%B0"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

    def test_url5(self):
        url="https://www.apiopen.top/meituApi?page=1"
        code,obj=self.get_url(url)
        self.assertEqual(code,200)
        self.assertEqual(obj['code'],200)

if __name__ == '__main__':
    unittest.main()