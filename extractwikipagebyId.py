import requests

def get_wiki_page_by_id(page_id: int):
    url = "https://vi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": page_id,
        "prop": "extracts|info|categories|pageimages|langlinks|extlinks|templates|links",
        "inprop": "url",
        "pithumbsize": 500,  # for thumbnails
        "explaintext": True,
        "cllimit": "max",
        "format": "json"
    }

    headers = {
        "User-Agent": "MyWikiFetcher/1.0 (contact: your_email@example.com)"
    }

    response = requests.get(url, params=params, headers=headers)

    # ✅ Check HTTP status first
    if response.status_code != 200:
        print(f"❌ HTTP error: {response.status_code}")
        print(response.text[:500])  # show first part of response
        exit()

    # ✅ Ensure response contains JSON
    if not response.text.strip().startswith("{"):
        print("❌ Response is not JSON (maybe blocked or empty):")
        print(response.text)
        exit()

    # Parse JSON safely
    data = response.json()

    # Trích xuất dữ liệu trang từ JSON
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page_data = next(iter(pages.values()))  # lấy phần tử đầu tiên
    return page_data


if __name__ == "__main__":
    page_id = 375318  # Ví dụ: trang "Internet Society"
    page = get_wiki_page_by_id(page_id)
    
    if page:
        print("✅ Tiêu đề:", page["title"])
        print("🔗 URL:", page["fullurl"])
        print("\n📜 Nội dung tóm tắt:\n")
        print(page["extract"][:800], "...")

        # ✅ Save to JSON file
        filename = f"wiki_page_{page_id}.json"
        import json
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=4)

        print(f"\n💾 Đã lưu dữ liệu vào: {filename}")

    else:
        print("❌ Không tìm thấy trang!")
