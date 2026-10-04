from playwright.sync_api import expect


def test_research_navigation_preserves_live_decision(page):
    events = page.locator('#revisionTimeline').inner_html()
    forecast = page.locator('#forecast').inner_text()
    target = page.locator('#targetTime').get_attribute('data-timestamp')
    page.locator('.engineering-evidence-head a[href="#researchEvidence"]').click()
    expect(page.locator('#technicalDetails')).to_have_attribute('open', '')
    expect(page.locator('#researchRows tr')).to_have_count(6)
    expect(page.locator('#researchVerdict')).to_contain_text('승격 거절')
    expect(page.locator('#researchVerdict')).to_contain_text('1 / 6')
    outage = page.locator('#researchRows tr').filter(has_text='Taylor/outage_2811')
    expect(outage).to_contain_text('+5.60%')
    expect(outage).to_contain_text('미달')
    assert page.locator('#forecast').inner_text() == forecast
    assert page.locator('#targetTime').get_attribute('data-timestamp') == target
    assert page.locator('#revisionTimeline').inner_html() == events
    assert page.locator('#showComparisonTruth').is_checked() is False


def test_research_fetch_failure_does_not_disable_model(page):
    page.locator('.engineering-evidence-head a[href="#researchEvidence"]').click()
    page.evaluate("window.failNext = '/static/research-evidence.json'")
    page.locator('#reloadResearch').click()
    expect(page.locator('#researchStatus')).to_contain_text('원문 보고서')
    expect(page.locator('#researchRows tr')).to_have_count(0)
    expect(page.locator('#researchVerdict')).to_have_text('판정 확인 중')
    page.locator('#previewChoices').click()
    expect(page.locator('#choiceComparison .choice-card')).to_have_count(3)
    page.locator('#reloadResearch').click()
    expect(page.locator('#researchRows tr')).to_have_count(6)


def test_expanded_research_and_evidence_links_fit_mobile(page):
    for width in [320, 390, 768]:
        page.set_viewport_size({'width': width, 'height': 1000})
        page.locator('#technicalDetails').evaluate('(element) => element.open = true')
        expect(page.locator('#researchRows tr')).to_have_count(6)
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
    links = page.locator('.engineering-evidence-grid a')
    assert links.count() == 6
    for link in links.all():
        assert link.get_attribute('href').startswith('https://github.com/sokldjs554/asofcast/')
    meta = page.evaluate("async () => JSON.parse((await asgiFetch('/api/metadata','GET',null)).body)")
    expect(page.locator('#runtimeScope')).to_contain_text(
        'ONNX Runtime' if meta['serving_backend'] == 'onnxruntime' else 'PyTorch'
    )
