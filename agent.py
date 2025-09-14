import datetime
from . import tools
from zoneinfo import ZoneInfo
from google.adk.agents import Agent,LlmAgent,BaseAgent
from google.adk.tools import agent_tool
from dotenv import load_dotenv
from typing import Dict, Any,Optional


import asyncio
import os

from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.planners import BasePlanner, BuiltInPlanner, PlanReActPlanner
from google.adk.models import LlmRequest

from google.genai.types import ThinkingConfig
from google.genai.types import GenerateContentConfig

import datetime
from zoneinfo import ZoneInfo
from uuid import UUID




# tools.py

import random

def get_merchant_status(restaurant_id: str) -> dict:
    return {
        "is_open": random.choice([True, False]),
        "staff_level": random.choice(["high", "medium", "low"]),
        "stock_level": random.choice(["adequate", "low", "empty"]),
        "queue_length": random.randint(0, 20)
    }

def check_weather(location: str) -> dict:
    return {
        "condition": random.choice(["rainy", "sunny", "storm"]),
        "severity": random.choice(["low", "medium", "high"]),
        "delivery_risk": random.choice(["low", "medium", "high"])
    }

def check_traffic(location: str) -> dict:
    return {
        "congestion_level": random.choice(["light", "moderate", "heavy"]),
        "avg_delay_minutes": random.randint(5, 60)
    }

def estimate_eta(start: str, destination: str) -> dict:
    return {"eta_minutes": random.randint(10, 60)}

def find_nearby_driver(location: str, radius_km: int) -> dict:
    return {
        "driver_id": f"driver_{random.randint(1000, 9999)}",
        "eta_minutes": random.randint(5, 20)
    }

def assign_driver(order_id: str, driver_id: str) -> dict:
    return {"success": random.choice([True, False])}

def reassign_order(order_id: str, driver_id: str) -> dict:
    return {"success": random.choice([True, False])}

def reschedule_delivery(order_id: str, new_time: str) -> dict:
    return {"success": random.choice([True, False])}

def cancel_order(order_id: str, reason: str) -> dict:
    return {"success": random.choice([True, False])}

def notify_customer(order_id: str, message: str) -> dict:
    return {"sent": random.choice([True, False])}


def offer_discount_coupon(order_id: str, coupon_code: str, discount_percent: int) -> dict:
    return {
        "success": random.choice([True, False])
    }
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


thinking_config = types.ThinkingConfig(
    include_thoughts=True,   # Ask the model to include its thoughts in the response
    thinking_budget=2048      # Limit the 'thinking' to 256 tokens (adjust as needed)
)
print("ThinkingConfig:", thinking_config)

# Step 2: Instantiate BuiltInPlanner
planner = BuiltInPlanner(
    thinking_config=thinking_config
)
traffic_agent = LlmAgent(
    name="traffic_agent",
    model="gemini-2.0-flash",
    description="Determines real-time traffic conditions and estimates travel time between specified locations.",
    instruction=(
        "Use traffic and weather data to assess road conditions and calculate estimated travel times "
        "between source and destination."
    ),
    tools=[check_traffic, check_weather, estimate_eta]
)

status_check_agent = LlmAgent(
    name="status_check_agent",
    model="gemini-2.0-flash",
    description="Manages all tasks related to checking the status of an order, including restaurant, traffic, and environmental conditions.",
    instruction=(
        "Check the status of the restaurant, weather, and traffic conditions using available tools. "
        "If a delay or issue is detected (e.g., restaurant not ready, bad traffic), "
        "determine whether to offer a compensation coupon or reschedule the delivery."
    ),
    tools=[
        get_merchant_status,
        agent_tool.AgentTool(agent=traffic_agent)  # Delegating traffic checks to traffic_agent
    ]
)

coupon_agent = LlmAgent(
    name="coupon_agent",
    model="gemini-2.0-flash",
    description="Handles logic for offering discount coupons and apology credits to customers.",
    instruction=(
        "Decide when to offer a discount or apology credit based on detected order or delivery issues. "
        "Execute the offer_discount_coupon tool when appropriate."
    ),
    tools=[offer_discount_coupon]
)

restaurant_planner = LlmAgent(
    name="restaurant_planner",
    model="gemini-2.0-flash",
    description=(
        "Main orchestrator agent responsible for managing user order status queries. "
        "It analyzes the user input, identifies missing information, and delegates tasks to specialized agents."
    ),
    instruction=(
        "When a user queries about order status or delay, determine which sub-agent to invoke based on the context. "
        "Interactively request missing data such as order ID or restaurant details if not provided by the user. "
        "Coordinate between status_check_agent, traffic_agent, and coupon_agent to resolve the issue. "
        "If needed, provide options for cancellation or compensation."
    ),
    sub_agents=[status_check_agent, traffic_agent, coupon_agent],
    tools=[cancel_order]
)

# ALL_TOOLS = [
#     verify_customer_presence,
#      contact_customer,
#      report_package_condition,
#      initiate_replacement,
#      process_refund,
#      offer_apology_credit,
#      offer_discount_coupon,
#      escalate_to_support,
#      upload_delivery_photo,
#      verify_delivery_photo,
#      upload_customer_complaint_photo,
#      verify_complaint_photo,
#  ] # or LlmAgent depending on your ADK import

user_related_issue = LlmAgent(
    name="user_related_issue",
    model="gemini-2.0-flash",
    description="Handles user-reported issues related to their order, such as product replacement requests, escalation to support, and reporting damaged or missing package conditions.",
    instruction=(
        "Analyze the user's complaint or query regarding their order. "
        "Decide whether to initiate a replacement, escalate the issue to support, "
        "or report the condition of the delivered package based on the context."
    ),
    tools=[
        initiate_replacement,
        escalate_to_support,
        report_package_condition
    ]
)

# driver_related_issue = LlmAgent(
#     name="driver_related_issue",
#     model="gemini-2.0-flash",
#     description="Manages issues related to delivery drivers, including verifying customer presence, contacting customers directly, and uploading delivery confirmation photos.",
#     instruction=(
#         "Process complaints or anomalies involving the delivery driver. "
#         "Decide to verify customer presence, contact the customer directly, "
#         "or upload photographic proof of delivery based on the situation."
#     ),
#     tools=[
#         verify_customer_presence,
#         contact_customer,
#         upload_delivery_photo
#     ]
# )

cancel_delivery_order = LlmAgent(
    name="cancel_delivery_order",
    model="gemini-2.0-flash",
    description="Handles order cancellation workflows, focusing on collecting evidence, verifying complaints, and providing customer compensations such as discount coupons or apology credits.",
    instruction=(
        "Coordinate the cancellation process by collecting supporting evidence, "
        "verifying complaint validity, and offering appropriate compensations "
        "to the customer to resolve the issue."
    ),
    tools=[
        upload_customer_complaint_photo,
        verify_complaint_photo,
        offer_discount_coupon,
        offer_apology_credit
    ]
)

door_step_planner = LlmAgent(
    name="door_step_planner",
    model="gemini-2.0-flash",
    description="Orchestrates the overall delivery issue resolution flow by intelligently selecting and coordinating relevant sub-agents based on the user's input.",
    instruction=(
        "Analyze the delivery issue reported by the customer. "
        "Decide whether the problem is user-related, driver-related, or requires order cancellation. "
        "Invoke the appropriate sub-agents to verify details, gather evidence, escalate to support, "
        "or provide compensations, ensuring the issue is efficiently resolved."
    ),
    sub_agents=[
        user_related_issue,
        # driver_related_issue,
        cancel_delivery_order
    ]
)
root_agent = LlmAgent(
    name="root_order_assistant",
    model="gemini-2.0-flash",
    description=(
        "Main orchestrator agent. Analyzes user query and delegates specific tasks "
        "to appropriate sub-agents like door_step_planner, restaurant_planner."
        "door_step_planner is related to returning misplaced or misfigured items."
        "restaurant_planner is related to restaurant_availability, order delivery,cancellation of food"
        
    ),
    instruction=(
        "call the tools based up on the query "
        "If any parameter is missing (like order_id), ask the user interactively."
    ),
    sub_agents=[restaurant_planner, door_step_planner]
)





#user_query,deliver_qurey,refund,