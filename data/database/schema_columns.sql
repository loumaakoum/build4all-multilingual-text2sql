-- categories
--   category_id integer NOT NULL
--   name character varying NOT NULL
--   parent_category_id integer NULL

-- coupons
--   coupon_id integer NOT NULL
--   code character varying NOT NULL
--   discount_type character varying NOT NULL
--   discount_value numeric NOT NULL
--   minimum_order_amount numeric NOT NULL
--   maximum_discount_amount numeric NULL
--   valid_from timestamp with time zone NULL
--   valid_to timestamp with time zone NULL
--   maximum_uses integer NULL
--   active boolean NOT NULL

-- customers
--   customer_id integer NOT NULL
--   customer_code character varying NOT NULL
--   city character varying NULL
--   region character varying NULL
--   created_at timestamp with time zone NOT NULL

-- order_items
--   order_item_id integer NOT NULL
--   order_id integer NOT NULL
--   product_id integer NOT NULL
--   quantity integer NOT NULL
--   unit_price numeric NOT NULL

-- orders
--   order_id integer NOT NULL
--   customer_id integer NOT NULL
--   order_date timestamp with time zone NOT NULL
--   status character varying NOT NULL
--   subtotal numeric NOT NULL
--   tax_amount numeric NOT NULL
--   shipping_amount numeric NOT NULL
--   discount_amount numeric NOT NULL
--   total_amount numeric NOT NULL
--   shipping_method_id integer NULL
--   coupon_id integer NULL

-- payments
--   payment_id integer NOT NULL
--   order_id integer NOT NULL
--   payment_method character varying NOT NULL
--   payment_status character varying NOT NULL
--   paid_amount numeric NOT NULL
--   payment_date timestamp with time zone NULL

-- product_events
--   event_id bigint NOT NULL
--   customer_id integer NULL
--   product_id integer NOT NULL
--   event_type character varying NOT NULL
--   event_time timestamp with time zone NOT NULL

-- products
--   product_id integer NOT NULL
--   sku character varying NOT NULL
--   name character varying NOT NULL
--   description text NULL
--   product_type character varying NOT NULL
--   price numeric NOT NULL
--   sale_price numeric NULL
--   stock integer NOT NULL
--   status character varying NOT NULL
--   category_id integer NULL
--   created_at timestamp with time zone NOT NULL

-- shipping_methods
--   shipping_method_id integer NOT NULL
--   name character varying NOT NULL
--   method_type character varying NOT NULL
--   cost numeric NOT NULL
--   country character varying NULL
--   region character varying NULL
--   active boolean NOT NULL

-- tax_rules
--   tax_rule_id integer NOT NULL
--   name character varying NOT NULL
--   rate_percent numeric NOT NULL
--   country character varying NULL
--   region character varying NULL
--   applies_to_shipping boolean NOT NULL
--   active boolean NOT NULL