import express from 'express';
import Stripe from 'stripe';
import cors from 'cors';
import dotenv from 'dotenv';
dotenv.config();

const app = express();
const PORT = process.env.PORT || 4242;
const STRIPE_SECRET = process.env.STRIPE_SECRET_KEY || 'sk_test_placeholder';
const stripe = new Stripe(STRIPE_SECRET);

const PRICES = {
  standard: process.env.STRIPE_PRICE_STANDARD || 'price_std',
  pro: process.env.STRIPE_PRICE_PRO || 'price_pro',
  elite: process.env.STRIPE_PRICE_ELITE || 'price_elite'
};

app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));
app.use(express.json());

// Create checkout session - supports sandbox mode when no Stripe key
app.post('/api/create-checkout-session', async (req, res) => {
  const { plan, email } = req.body;
  
  if (!plan || !['standard', 'pro', 'elite'].includes(plan)) {
    return res.status(400).json({ error: 'Invalid plan. Use standard, pro, or elite' });
  }

  if (STRIPE_SECRET.includes('placeholder')) {
    console.log(`[SANDBOX] Creating checkout for ${plan} / ${email}`);
    return res.json({
      sandbox: true,
      plan,
      url: `/?checkout=success&plan=${plan}&session=sandbox_${Date.now()}`
    });
  }

  try {
    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      customer_email: email,
      line_items: [{ price: PRICES[plan], quantity: 1 }],
      success_url: `${process.env.FRONTEND_URL}/?checkout=success&plan=${plan}&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.FRONTEND_URL}/?checkout=cancel`,
      metadata: { plan },
      allow_promotion_codes: true,
      billing_address_collection: 'auto'
    });
    res.json({ url: session.url, id: session.id });
  } catch (e) {
    console.error('Stripe error', e);
    res.status(500).json({ error: e.message });
  }
});

// Verify session after redirect
app.get('/api/verify-session', async (req, res) => {
  const { session_id } = req.query;
  if (!session_id) return res.status(400).json({ error: 'Missing session_id' });

  if (String(session_id).startsWith('sandbox_')) {
    return res.json({
      sandbox: true,
      plan: req.query.plan || 'pro',
      status: 'complete',
      customer_email: 'sandbox@echoforge.ai'
    });
  }

  try {
    const s = await stripe.checkout.sessions.retrieve(session_id);
    res.json({
      status: s.payment_status,
      plan: s.metadata?.plan,
      customer_email: s.customer_details?.email
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/health', (req, res) => res.json({ ok: true, timestamp: Date.now(), stripe_configured: !STRIPE_SECRET.includes('placeholder') }));

// Optional webhook for production
app.post('/api/webhook', express.raw({ type: 'application/json' }), (req, res) => {
  // Implement stripe.webhooks.constructEvent with STRIPE_WEBHOOK_SECRET
  res.json({ received: true });
});

app.listen(PORT, () => console.log(`[EchoForge] Backend running on :${PORT} | ${STRIPE_SECRET.includes('placeholder') ? 'SANDBOX MODE' : 'LIVE'}`));
