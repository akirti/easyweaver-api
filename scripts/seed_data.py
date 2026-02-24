"""Seed Postgres and MongoDB with demo data, then register sources and a demo user.

Usage:
  poetry run python scripts/seed_data.py              # For local dev (API on host)
  poetry run python scripts/seed_data.py --docker     # For Docker (API in container)
"""

import asyncio
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import bcrypt
from cryptography.fernet import Fernet
from motor.motor_asyncio import AsyncIOMotorClient

# When --docker flag is passed, registered sources use Docker service names
# so the API container can connect to them via the Docker network
DOCKER_MODE = "--docker" in sys.argv

# Connection strings for the seed script (always runs on the host)
PG_DSN = os.environ.get(
    "SEED_PG_DSN", "postgresql://easyweaver:easyweaver@localhost:5432/easyweaver"
)
MONGO_URI = os.environ.get("SEED_MONGO_URI", "mongodb://easyweaver:easyweaver@localhost:27018")
MONGO_DEMO_DB = "easyweaver_demo"
MONGO_META_DB = "easyweaver_meta"

NUM_USERS = 1000
NUM_PRODUCTS = 500
NUM_ORDERS = 5000

FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
CATEGORIES = ["Electronics", "Clothing", "Books", "Home", "Sports", "Toys", "Food", "Health"]
STATUSES = ["pending", "confirmed", "shipped", "delivered", "cancelled"]

# Must match EW_FERNET_KEY in settings / .env
FERNET_KEY = None  # Will be generated if not set


def get_fernet_key():
    """Get or generate a Fernet key."""
    global FERNET_KEY
    if FERNET_KEY is None:
        # Try reading from .env
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("EW_FERNET_KEY="):
                        key = line.strip().split("=", 1)[1].strip("\"'")
                        if key and key != "change-me-generate-with-cryptography-fernet":
                            FERNET_KEY = key
                            return FERNET_KEY
        except FileNotFoundError:
            pass

        # Generate a new key and write to .env
        FERNET_KEY = Fernet.generate_key().decode()
        print(f"Generated new Fernet key: {FERNET_KEY}")

        # Append to .env file
        with open(".env", "a") as f:
            f.write(f"\nEW_FERNET_KEY={FERNET_KEY}\n")
        print("Saved Fernet key to .env")

    return FERNET_KEY


def encrypt_credentials(creds_json: str) -> str:
    """Encrypt credentials using the Fernet key."""
    f = Fernet(get_fernet_key().encode())
    return f.encrypt(creds_json.encode()).decode()


async def seed_postgres():
    conn = await asyncpg.connect(PG_DSN)

    # Create tables
    await conn.execute("""
        DROP TABLE IF EXISTS orders CASCADE;
        DROP TABLE IF EXISTS products CASCADE;
        DROP TABLE IF EXISTS demo_users CASCADE;
    """)

    await conn.execute("""
        CREATE TABLE demo_users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            age INTEGER,
            city VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    await conn.execute("""
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            category VARCHAR(100) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            stock INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    await conn.execute("""
        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES demo_users(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL DEFAULT 1,
            total_amount DECIMAL(10,2) NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            order_date TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Seed users
    users = []
    for i in range(1, NUM_USERS + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        users.append((
            f"{first.lower()}.{last.lower()}{i}@example.com",
            first,
            last,
            random.randint(18, 70),
            random.choice(["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney"]),
        ))

    await conn.executemany(
        "INSERT INTO demo_users (email, first_name, last_name, age, city) VALUES ($1,$2,$3,$4,$5)",
        users,
    )

    # Seed products
    products = []
    for i in range(1, NUM_PRODUCTS + 1):
        cat = random.choice(CATEGORIES)
        products.append((
            f"{cat} Item {i}",
            cat,
            round(random.uniform(5.0, 500.0), 2),
            random.randint(0, 1000),
        ))

    await conn.executemany(
        "INSERT INTO products (name, category, price, stock) VALUES ($1,$2,$3,$4)",
        products,
    )

    # Seed orders in Postgres too (for cross-db join demos)
    base_date = datetime(2024, 1, 1)
    pg_orders = []
    for _ in range(NUM_ORDERS):
        user_id = random.randint(1, NUM_USERS)
        product_id = random.randint(1, NUM_PRODUCTS)
        quantity = random.randint(1, 10)
        total = round(random.uniform(10.0, 2000.0), 2)
        status = random.choice(STATUSES)
        order_date = base_date + timedelta(days=random.randint(0, 365))
        pg_orders.append((user_id, product_id, quantity, total, status, order_date))

    await conn.executemany(
        "INSERT INTO orders (user_id, product_id, quantity, total_amount, status, order_date) "
        "VALUES ($1,$2,$3,$4,$5,$6)",
        pg_orders,
    )

    await conn.close()
    print(f"Postgres: seeded {NUM_USERS} users, {NUM_PRODUCTS} products, {NUM_ORDERS} orders")


async def seed_mongodb():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DEMO_DB]

    # Drop existing
    await db.orders.drop()
    await db.user_profiles.drop()
    await db.support_tickets.drop()

    # Seed orders (cross-referenceable with Postgres users via user_id)
    orders = []
    base_date = datetime(2024, 1, 1)
    for _ in range(NUM_ORDERS):
        user_id = random.randint(1, NUM_USERS)
        product_id = random.randint(1, NUM_PRODUCTS)
        quantity = random.randint(1, 10)
        order_date = base_date + timedelta(days=random.randint(0, 365))
        orders.append({
            "order_id": str(uuid.uuid4()),
            "user_id": user_id,
            "product_id": product_id,
            "quantity": quantity,
            "total_amount": round(random.uniform(10.0, 2000.0), 2),
            "status": random.choice(STATUSES),
            "order_date": order_date,
            "shipping_address": {
                "street": f"{random.randint(1, 999)} Main St",
                "city": random.choice(["New York", "London", "Tokyo", "Paris"]),
                "country": random.choice(["US", "UK", "JP", "FR"]),
            },
        })

    await db.orders.insert_many(orders)

    # Seed user profiles (extra MongoDB-native data, cross-ref with Postgres users)
    profiles = []
    for i in range(1, NUM_USERS + 1):
        profiles.append({
            "user_id": i,
            "preferences": {
                "theme": random.choice(["light", "dark"]),
                "notifications": random.choice([True, False]),
                "language": random.choice(["en", "fr", "de", "ja"]),
            },
            "tags": random.sample(
                ["vip", "new", "returning", "wholesale", "premium"], k=random.randint(0, 3)
            ),
            "last_login": base_date + timedelta(days=random.randint(0, 365)),
        })

    await db.user_profiles.insert_many(profiles)

    # Seed support tickets (nested data for MongoDB querying demos)
    tickets = []
    priorities = ["low", "medium", "high", "critical"]
    ticket_statuses = ["open", "in_progress", "resolved", "closed"]
    for _ in range(2000):
        user_id = random.randint(1, NUM_USERS)
        created = base_date + timedelta(days=random.randint(0, 365))
        tickets.append({
            "ticket_id": str(uuid.uuid4())[:8].upper(),
            "user_id": user_id,
            "subject": random.choice([
                "Cannot login", "Order not received", "Refund request",
                "Product damaged", "Account settings", "Payment issue",
                "Shipping delay", "Wrong item received",
            ]),
            "priority": random.choice(priorities),
            "status": random.choice(ticket_statuses),
            "created_at": created,
            "resolved_at": created + timedelta(days=random.randint(1, 30))
            if random.random() > 0.3
            else None,
            "metadata": {
                "channel": random.choice(["email", "chat", "phone"]),
                "agent_id": random.randint(1, 20),
            },
        })

    await db.support_tickets.insert_many(tickets)

    client.close()
    print(
        f"MongoDB: seeded {NUM_ORDERS} orders, {NUM_USERS} user profiles, "
        f"{len(tickets)} support tickets"
    )


async def register_sources_and_user():
    """Register demo data sources and a demo user in easyweaver_meta."""
    client = AsyncIOMotorClient(MONGO_URI)
    meta_db = client[MONGO_META_DB]
    now = datetime.now(timezone.utc)

    # Clear existing demo sources and users
    await meta_db.data_sources.delete_many({"name": {"$regex": "^Demo"}})
    await meta_db.users.delete_many({"email": "demo@easyweaver.io"})

    # --- Register demo user ---
    hashed_pw = bcrypt.hashpw("demo1234".encode(), bcrypt.gensalt()).decode()
    demo_user = {
        "_id": str(uuid.uuid4()),
        "email": "demo@easyweaver.io",
        "hashed_password": hashed_pw,
        "display_name": "Demo User",
        "role": "admin",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    await meta_db.users.insert_one(demo_user)
    print(f"Registered demo user: demo@easyweaver.io / demo1234")

    # --- Register Postgres data source ---
    pg_host = "postgres" if DOCKER_MODE else "localhost"
    pg_creds = json.dumps({
        "type": "postgres",
        "host": pg_host,
        "port": 5432,
        "database": "easyweaver",
        "user": "easyweaver",
        "password": "easyweaver",
    })
    pg_source = {
        "_id": str(uuid.uuid4()),
        "name": "Demo Postgres",
        "source_type": "postgres",
        "encrypted_credentials": encrypt_credentials(pg_creds),
        "metadata": None,
        "created_at": now,
        "updated_at": now,
    }
    await meta_db.data_sources.insert_one(pg_source)
    print(f"Registered source: Demo Postgres (tables: demo_users, products, orders)")

    # --- Register MongoDB data source ---
    mongo_host = "mongodb" if DOCKER_MODE else "localhost"
    mongo_port = 27017 if DOCKER_MODE else 27018
    mongo_creds = json.dumps({
        "type": "mongodb",
        "host": mongo_host,
        "port": mongo_port,
        "database": "easyweaver_demo",
        "user": "easyweaver",
        "password": "easyweaver",
        "auth_database": "admin",
    })
    mongo_source = {
        "_id": str(uuid.uuid4()),
        "name": "Demo MongoDB",
        "source_type": "mongodb",
        "encrypted_credentials": encrypt_credentials(mongo_creds),
        "metadata": None,
        "created_at": now,
        "updated_at": now,
    }
    await meta_db.data_sources.insert_one(mongo_source)
    print(f"Registered source: Demo MongoDB (collections: orders, user_profiles, support_tickets)")

    client.close()


async def main():
    print("=" * 60)
    print("EasyWeaver — Seeding Demo Data")
    print("=" * 60)

    print("\n[1/3] Seeding Postgres...")
    await seed_postgres()

    print("\n[2/3] Seeding MongoDB...")
    await seed_mongodb()

    print("\n[3/3] Registering sources and demo user...")
    await register_sources_and_user()

    print("\n" + "=" * 60)
    print("Seed complete!")
    print("=" * 60)
    print("\nLogin credentials:")
    print("  Email:    demo@easyweaver.io")
    print("  Password: demo1234")
    print("\nData sources registered:")
    print("  Demo Postgres  — demo_users, products, orders")
    print("  Demo MongoDB   — orders, user_profiles, support_tickets")
    print("\nCross-source join keys:")
    print("  Postgres demo_users.id  <->  MongoDB orders.user_id")
    print("  Postgres demo_users.id  <->  MongoDB user_profiles.user_id")
    print("  Postgres demo_users.id  <->  MongoDB support_tickets.user_id")
    print("  Postgres products.id    <->  MongoDB orders.product_id")


if __name__ == "__main__":
    asyncio.run(main())
