# delivery_tools.py
import random
import uuid
import datetime
from typing import Dict, Any, Optional
from google.adk.agents import LlmAgent 
# ADK FunctionTool wrapper

# -------------------------
# Mock stores / helpers
# -------------------------
_photo_store = {}
_complaint_store = {}
_ticket_store = {}
_contacts_attempts = {}

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _gen_url(kind: str, order_id: str) -> str:
    return f"https://mock.storage/{kind}/{order_id}/{uuid.uuid4().hex}.jpg"

def _gen_ticket(issue_type: str) -> str:
    ticket = f"TKT-{issue_type[:3].upper()}-{random.randint(10000,99999)}"
    _ticket_store[ticket] = {"issue_type": issue_type, "created_at": _now_iso()}
    return ticket

# -------------------------
# Tools implementations
# -------------------------

def verify_customer_presence(order_id: str) -> Dict[str, Any]:
    """
    Check if customer is present at delivery address (mock).
    Returns: { present: bool, last_contact_attempt: ISO timestamp or None }
    """
    # simulate previous attempts
    attempts = _contacts_attempts.get(order_id, [])
    last_attempt = attempts[-1]["timestamp"] if attempts else None
    present = random.choice([True, False, False])  # bias to absent
    return {"status": "success", "present": bool(present), "last_contact_attempt": last_attempt}

def contact_customer(order_id: str, method: str = "call") -> Dict[str, Any]:
    """
    Attempt to contact customer. method in {"call","sms","app"}.
    Returns: { success: bool, response: "reschedule"|"pickup"|"cancel"|None }
    """
    method = method.lower()
    possible = ["reschedule", "pickup", "cancel", None]
    # simulate success and response
    success = random.choice([True, True, False])
    response = random.choice(possible) if success else None
    attempt = {"method": method, "success": success, "response": response, "timestamp": _now_iso()}
    _contacts_attempts.setdefault(order_id, []).append(attempt)
    return {"status": "success", "success": success, "response": response, "attempt": attempt}

def leave_package_at_safe_spot(order_id: str, location: str) -> Dict[str, Any]:
    """
    Leave package at a safe spot and produce photo proof (mock).
    Returns: { success: bool, photo_proof_url, timestamp }
    """
    success = random.choice([True, True, False])
    photo_url = _gen_url("delivery_proofs", order_id) if success else None
    if success:
        _photo_store.setdefault(order_id, []).append({"type": "delivery_proof", "url": photo_url, "location": location, "timestamp": _now_iso()})
    return {"status": "success", "success": success, "photo_proof_url": photo_url, "timestamp": _now_iso(), "location": location}

def report_package_condition(order_id: str, condition: str) -> Dict[str, Any]:
    """
    Courier reports package condition: "damaged"|"leaked"|"ok".
    Returns: { condition, severity (1-5), note, timestamp }
    """
    condition = condition.lower()
    severity_map = {"ok": 1, "leaked": 4, "damaged": 5}
    severity = severity_map.get(condition, random.randint(1,5))
    note = {
        "ok": "Package intact",
        "leaked": "Liquid observed, minor leak",
        "damaged": "Box crushed or torn"
    }.get(condition, "Reported condition")
    return {"status": "success", "condition": condition, "severity": severity, "note": note, "timestamp": _now_iso()}

def initiate_replacement(order_id: str) -> Dict[str, Any]:
    """
    Start replacement flow.
    Returns: { success: bool, new_eta: minutes_from_now (int) or None, replacement_id }
    """
    success = random.choice([True, False, True])
    new_eta = random.randint(30, 180) if success else None
    replacement_id = f"rep_{uuid.uuid4().hex[:8]}" if success else None
    return {"status": "success", "success": success, "new_eta_minutes": new_eta, "replacement_id": replacement_id}

def process_refund(order_id: str, amount: float) -> Dict[str, Any]:
    """
    Process monetary refund (mock).
    Returns: { success: bool, refunded_amount: float or 0, txn_id or None }
    """
    success = random.choice([True, True, False])
    txn_id = f"txn_{uuid.uuid4().hex[:10]}" if success else None
    refunded = amount if success else 0.0
    return {"status": "success", "success": success, "refunded_amount": refunded, "txn_id": txn_id}

def offer_apology_credit(order_id: str, amount: float) -> Dict[str, Any]:
    """
    Offer apology credit to customer account.
    Returns: { success: bool, credited_amount, credit_id }
    """
    success = random.choice([True, True, False])
    credit_id = f"credit_{uuid.uuid4().hex[:8]}" if success else None
    return {"status": "success", "success": success, "credited_amount": amount if success else 0.0, "credit_id": credit_id}

def offer_discount_coupon(order_id: str, code: str, discount_percent: int) -> Dict[str, Any]:
    """
    Offer a discount coupon; returns whether applied/sent.
    """
    applied = random.choice([True, False, True])
    return {"status": "success", "success": applied, "coupon_code": code, "discount_percent": discount_percent}

def escalate_to_support(order_id: str, issue_type: str) -> Dict[str, Any]:
    """
    Escalate issue to support -> returns ticket_id.
    """
    ticket_id = _gen_ticket(issue_type)
    return {"status": "success", "ticket_id": ticket_id, "created_at": _now_iso(), "issue_type": issue_type}

def upload_delivery_photo(order_id: str, photo_url: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Store a delivered photo proof into mock storage.
    Returns: { success, stored_at (url), timestamp }
    """
    stored_at = _gen_url("uploaded_photos", order_id)
    _photo_store.setdefault(order_id, []).append({"type": "delivery", "orig_url": photo_url, "stored_at": stored_at, "context": context, "timestamp": _now_iso()})
    return {"status": "success", "success": True, "stored_at": stored_at, "timestamp": _now_iso()}

def verify_delivery_photo(order_id: str) -> Dict[str, Any]:
    """
    Verify last delivery photo for evidence of safe delivery. Returns confidence score.
    """
    photos = _photo_store.get(order_id, [])
    if not photos:
        return {"status": "error", "verified": False, "confidence_score": 0.0, "reason": "no_photos"}
    # mock verification
    confidence = round(random.uniform(0.5, 0.99), 2)
    verified = confidence > 0.7
    return {"status": "success", "verified": bool(verified), "confidence_score": confidence, "checked_photos": len(photos)}

def upload_customer_complaint_photo(order_id: str, photo_url: str, issue_type: str) -> Dict[str, Any]:
    """
    Upload customer's complaint photo (damaged/tampered/etc)
    """
    stored_at = _gen_url("complaint_photos", order_id)
    entry = {"order_id": order_id, "orig_url": photo_url, "stored_at": stored_at, "issue_type": issue_type, "timestamp": _now_iso()}
    _complaint_store.setdefault(order_id, []).append(entry)
    return {"status": "success", "success": True, "stored_at": stored_at, "issue_type": issue_type, "timestamp": _now_iso()}

def verify_complaint_photo(order_id: str) -> Dict[str, Any]:
    """
    Verify customer's complaint photo and return condition.
    """
    complaints = _complaint_store.get(order_id, [])
    if not complaints:
        return {"status": "error", "verified": False, "condition": "ok", "reason": "no_complaint_photos"}
    confidence = round(random.uniform(0.4, 0.99), 2)
    # mock classification
    condition = random.choices(["damaged", "tampered", "ok"], weights=[0.4, 0.2, 0.4])[0]
    verified = confidence > 0.6 and condition != "ok"
    return {"status": "success", "verified": bool(verified), "condition": condition, "confidence_score": confidence, "checked_photos": len(complaints)}

# -------------------------
# FunctionTool wrappers
# -------------------------

# Optionally expose a dict for quick registration
ALL_TOOLS = [
    verify_customer_presence,
     contact_customer,
     leave_package_at_safe_spot,
     report_package_condition,
     initiate_replacement,
     process_refund,
     offer_apology_credit,
     offer_discount_coupon,
     escalate_to_support,
     upload_delivery_photo,
     verify_delivery_photo,
     upload_customer_complaint_photo,
     verify_complaint_photo,
 ] # or LlmAgent depending on your ADK import

#user_query,deliver_qurey,refund,
my_agent = LlmAgent(
    name="door-step-planner",
    model="gemini-2.0-flash",
    description="ochrestate the flow .by calling the agents..",
    instruction="Use available tools to verify delivery and escalate or resolve issues.",
    tools=ALL_TOOLS  # or select subset
)
