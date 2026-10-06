-- Build4All research database (synthetic data only).
-- Column names and types follow schema_columns.sql, which was exported from the live schema.
CREATE SCHEMA IF NOT EXISTS build4all_research;
SET search_path TO build4all_research;

CREATE TABLE categories (
  category_id integer PRIMARY KEY,
  name varchar NOT NULL,
  parent_category_id integer REFERENCES categories(category_id)
);
CREATE TABLE customers (
  customer_id integer PRIMARY KEY,
  customer_code varchar NOT NULL,
  city varchar,
  region varchar,
  created_at timestamptz NOT NULL
);
CREATE TABLE products (
  product_id integer PRIMARY KEY,
  sku varchar NOT NULL,
  name varchar NOT NULL,
  description text,
  product_type varchar NOT NULL,
  price numeric NOT NULL,
  sale_price numeric,
  stock integer NOT NULL,
  status varchar NOT NULL,
  category_id integer REFERENCES categories(category_id),
  created_at timestamptz NOT NULL
);
CREATE TABLE shipping_methods (
  shipping_method_id integer PRIMARY KEY,
  name varchar NOT NULL,
  method_type varchar NOT NULL,
  cost numeric NOT NULL,
  country varchar,
  region varchar,
  active boolean NOT NULL
);
CREATE TABLE coupons (
  coupon_id integer PRIMARY KEY,
  code varchar NOT NULL,
  discount_type varchar NOT NULL,
  discount_value numeric NOT NULL,
  minimum_order_amount numeric NOT NULL,
  maximum_discount_amount numeric,
  valid_from timestamptz,
  valid_to timestamptz,
  maximum_uses integer,
  active boolean NOT NULL
);
CREATE TABLE tax_rules (
  tax_rule_id integer PRIMARY KEY,
  name varchar NOT NULL,
  rate_percent numeric NOT NULL,
  country varchar,
  region varchar,
  applies_to_shipping boolean NOT NULL,
  active boolean NOT NULL
);
CREATE TABLE orders (
  order_id integer PRIMARY KEY,
  customer_id integer NOT NULL REFERENCES customers(customer_id),
  order_date timestamptz NOT NULL,
  status varchar NOT NULL,
  subtotal numeric NOT NULL,
  tax_amount numeric NOT NULL,
  shipping_amount numeric NOT NULL,
  discount_amount numeric NOT NULL,
  total_amount numeric NOT NULL,
  shipping_method_id integer REFERENCES shipping_methods(shipping_method_id),
  coupon_id integer REFERENCES coupons(coupon_id)
);
CREATE TABLE order_items (
  order_item_id integer PRIMARY KEY,
  order_id integer NOT NULL REFERENCES orders(order_id),
  product_id integer NOT NULL REFERENCES products(product_id),
  quantity integer NOT NULL,
  unit_price numeric NOT NULL
);
CREATE TABLE payments (
  payment_id integer PRIMARY KEY,
  order_id integer NOT NULL REFERENCES orders(order_id),
  payment_method varchar NOT NULL,
  payment_status varchar NOT NULL,
  paid_amount numeric NOT NULL,
  payment_date timestamptz
);
CREATE TABLE product_events (
  event_id bigint PRIMARY KEY,
  customer_id integer REFERENCES customers(customer_id),
  product_id integer NOT NULL REFERENCES products(product_id),
  event_type varchar NOT NULL,
  event_time timestamptz NOT NULL
);
