"""Seed Postgres and MongoDB with demo data for cross-source join testing."""

import asyncio
import random
import uuid
from datetime import datetime, timedelta

import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient


PG_DSN = "postgresql://easyweaver:easyweaver@localhost:5432/easyweaver"
MONGO_URI = "mongodb://easyweaver:easyweaver@localhost:27018"
MONGO_DB = "easyweaver_demo"

NUM_USERS = 1000
NUM_PRODUCTS = 500
NUM_ORDERS = 5000

FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
CATEGORIES = ["Electronics", "Clothing", "Books", "Home", "Sports", "Toys", "Food", "Health"]
STATUSES = ["pending", "confirmed", "shipped", "delivered", "cancelled"]


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

    await conn.close()
    print(f"Postgres: seeded {NUM_USERS} users, {NUM_PRODUCTS} products")


async def seed_mongodb():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]

    # Drop existing
    await db.orders.drop()
    await db.user_profiles.drop()

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
            "tags": random.sample(["vip", "new", "returning", "wholesale", "premium"], k=random.randint(0, 3)),
            "last_login": base_date + timedelta(days=random.randint(0, 365)),
        })

    await db.user_profiles.insert_many(profiles)

    client.close()
    print(f"MongoDB: seeded {NUM_ORDERS} orders, {NUM_USERS} user profiles")


async def main():
    print("Seeding demo data...")
    await seed_postgres()
    await seed_mongodb()
    print("Done! Ready for cross-source join demos.")
    print("  Postgres: demo_users (id) <-> MongoDB: orders (user_id)")
    print("  Postgres: products (id) <-> MongoDB: orders (product_id)")


if __name__ == "__main__":
    asyncio.run(main())
