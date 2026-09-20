"""Resolve editable controls by identity, never by their ordinal among clickable nodes."""
import json

ROW_IDENTITY_JS = """el => {
 const row = el.closest('tr, [role="row"]');
 if (!row) return null;
 for (const name of ['data-row-id', 'data-id', 'id']) {
   const value = row.getAttribute(name);
   if (value) return {kind: 'attribute', name, value};
 }
 const boxes = row.querySelectorAll('input[type="checkbox"][value]');
 if (boxes.length === 1 && boxes[0].value && boxes[0].value !== 'on')
   return {kind: 'checkbox', value: boxes[0].value};
 return {kind: 'unidentified'};
}"""


def capture_row_identity(locator):
    return locator.evaluate(ROW_IDENTITY_JS)


def input_selector(element):
    tag = element.tag
    if tag not in ('input', 'textarea'):
        raise ValueError('Expected input or textarea')
    selector = tag
    attrs = getattr(element, 'attributes_dict', {}) or {}
    for attr, field in [('id','id'), ('name','name'), ('data-testid','data_testid'),
                        ('aria-label','aria_label'), ('placeholder','placeholder')]:
        value = getattr(element, field, None) or attrs.get(attr)
        if value:
            selector += '['+attr+'='+json.dumps(str(value),ensure_ascii=False)+']'
    if tag == 'input':
        typ = getattr(element, 'input_type', None) or attrs.get('type') or 'text'
        selector += ':is([type="text"],:not([type]))' if typ == 'text' else '[type='+json.dumps(typ)+']'
    for cls in str(getattr(element, 'class_name', '') or '').split():
        selector += '[class~='+json.dumps(cls,ensure_ascii=False)+']'
    return selector


def resolve_input(base, element):
    identity = getattr(element, 'row_identity', None)
    if identity:
        rows = base.locator('tr, [role="row"]')
        if identity['kind'] == 'attribute':
            rows = rows.and_(base.locator('['+identity['name']+'='+json.dumps(identity['value'])+']'))
        elif identity['kind'] == 'checkbox':
            rows = rows.filter(has=base.page.locator('input[type="checkbox"][value='+json.dumps(identity['value'])+']'))
        else:
            return base.locator(':not(*)')
        if rows.count() != 1:
            return base.locator(':not(*)')
        base = rows
    return base.locator(input_selector(element)).filter(visible=True)


def validated_input_handle(locator, element):
    """Pin the validated DOM node so a later reorder cannot retarget the fill."""
    if locator.count() != 1:
        return None
    handle = locator.element_handle()
    if handle is None:
        return None
    expected = getattr(element, 'row_identity', None)
    matches = handle.evaluate('(el, selector) => el.isConnected && el.matches(selector)', input_selector(element))
    if expected and handle.evaluate(ROW_IDENTITY_JS) != expected:
        matches = False
    typ = handle.get_attribute('type') or 'text'
    if not matches or typ in ('checkbox','radio','file','button','submit','reset','image','hidden','range','color') or not handle.is_editable():
        handle.dispose()
        return None
    return handle
