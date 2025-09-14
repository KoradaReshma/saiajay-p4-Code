import random
from datetime import datetime, timedelta

# ------------------------------
# Sample Mock Data
# ------------------------------

MERCHANTS = {
    "rest_101": { "is_open": True, "staff_level": "high", "stock_level": "medium", "queue_length": 15 },
    "rest_102": { "is_open": False, "staff_level": "low", "stock_level": "low", "queue_length": 0 },
    "rest_103": { "is_open": True, "staff_level": "medium", "stock_level": "low", "queue_length": 30 },
}

WEATHER = {
    "New York": { "condition": "rainy", "severity": "moderate", "delivery_risk": "high" },
    "San Francisco": { "condition": "sunny", "severity": "low", "delivery_risk": "low" },
    "Chicago": { "condition": "storm", "severity": "high", "delivery_risk": "very_high" }
}

TRAFFIC = {
    "Downtown": { "congestion_level": "heavy", "avg_delay_minutes": 25 },
    "Uptown": { "congestion_level": "moderate", "avg_delay_minutes": 10 },
    "Suburbs": { "congestion_level": "light", "avg_delay_minutes": 3 }
}

DRIVERS = [
    { "driver_id": "drv_001", "location": "Downtown", "eta_minutes": 5 },
    { "driver_id": "drv_002", "location": "Uptown", "eta_minutes": 12 },
    { "driver_id": "drv_003", "location": "Suburbs", "eta_minutes": 20 },
]

ORDERS = {
    "ord_1001": { "restaurant_id": "rest_101", "customer_location": "Downtown", "status": "pending" },
    "ord_1002": { "restaurant_id": "rest_103", "customer_location": "Suburbs", "status": "pending" }
}

COUPONS = [
    { "coupon_code": "DISC10", "discount_percent": 10 },
    { "coupon_code": "DISC20", "discount_percent": 20 }
]

COMPLEMENTARY_ITEMS = [
    { "item_id": "comp_001", "description": "Free dessert" },
    { "item_id": "comp_002", "description": "Free soft drink" }
]

# ------------------------------
# Tool Implementations
# ------------------------------

def get_merchant_status(restaurant_id):
    return MERCHANTS.get(restaurant_id, {"error": "Restaurant not found"})


def check_weather(location):
    return WEATHER.get(location, {"condition": "unknown", "severity": "unknown", "delivery_risk": "unknown"})


def check_traffic(location):
    return TRAFFIC.get(location, {"congestion_level": "unknown", "avg_delay_minutes": -1})


def estimate_eta(start, destination):
    base_eta = random.randint(10, 30)
    traffic_info = check_traffic(destination)
    delay = traffic_info.get("avg_delay_minutes", 0)
    eta = base_eta + delay
    return { "eta_minutes": eta }


def find_nearby_driver(location, radius_km):
    # Simple simulation: pick first available driver in the same location
    for driver in DRIVERS:
        if driver["location"] == location:
            return { "driver_id": driver["driver_id"], "eta_minutes": driver["eta_minutes"] }
    return { "driver_id": None, "eta_minutes": -1 }


def assign_driver(order_id, driver_id):
    order = ORDERS.get(order_id)
    if order:
        order["driver_id"] = driver_id
        order["status"] = "assigned"
        return { "success": True }
    return { "success": False }


def reassign_order(order_id, driver_id):
    return assign_driver(order_id, driver_id)


def reschedule_delivery(order_id, new_time):
    order = ORDERS.get(order_id)
    if order:
        order["rescheduled_time"] = new_time
        return { "success": True }
    return { "success": False }


def cancel_order(order_id, reason):
    order = ORDERS.get(order_id)
    if order:
        order["status"] = "cancelled"
        order["cancellation_reason"] = reason
        return { "success": True }
    return { "success": False }


def notify_customer(order_id, message):
    # Simulate notification success
    print(f"Notification to customer of {order_id}: {message}")
    return { "sent": True }


def offer_complementary_item(order_id, item_id):
    item = next((item for item in COMPLEMENTARY_ITEMS if item["item_id"] == item_id), None)
    if item:
        return { "success": True, "message": f"Offered complementary item: {item['description']}" }
    return { "success": False, "message": "Item not found" }


def offer_discount_coupon(order_id, coupon_code, discount_percent):
    coupon = next((c for c in COUPONS if c["coupon_code"] == coupon_code and c["discount_percent"] == discount_percent), None)
    if coupon:
        return { "success": True, "message": f"Applied coupon {coupon_code} for {discount_percent}% discount" }
    return { "success": False, "message": "Coupon not valid" }
