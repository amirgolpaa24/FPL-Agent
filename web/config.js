// FPL Agent frontend configuration.
// To enable accounts (free): create a Supabase project, run supabase/schema.sql,
// then paste your project's URL + anon key here (Settings -> API).
// The ANON key is safe to ship in the frontend - Row Level Security protects data.
// Leave both empty to run without accounts (everything still works).
window.FPL_CONFIG = {
  SUPABASE_URL: "",        // e.g. "https://abcdefgh.supabase.co"
  SUPABASE_ANON_KEY: "",   // e.g. "eyJhbGciOi..."
};
