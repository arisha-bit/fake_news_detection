from fastapi import HTTPException


def test_reverse_image_search_falls_back_when_index_missing(monkeypatch, tmp_path):
    import app.services.image_service as image_service

    def fake_get_clip():
        return object(), object()

    def fake_get_index():
        raise HTTPException(status_code=503, detail="missing index")

    monkeypatch.setattr(image_service, "_get_clip", fake_get_clip)
    monkeypatch.setattr(image_service, "_get_index", fake_get_index)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not-a-real-image")

    results, possible_reuse = image_service.reverse_image_search(str(image_path), top_k=3)

    assert results == []
    assert possible_reuse is False
