import pg from "pg";

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS detections (
    sha256         bytea            NOT NULL,
    model          text             NOT NULL,
    logit          double precision NOT NULL,
    created_at     timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (sha256, model)
  )
`;

export class DetectionStore {
  readonly pool: pg.Pool;

  constructor(pool: pg.Pool) {
    this.pool = pool;
  }

  async migrate(): Promise<void> {
    await this.pool.query(SCHEMA);
  }

  async find(sha256: Buffer, model: string): Promise<number | undefined> {
    const { rows } = await this.pool.query<{ logit: number }>(
      "SELECT logit FROM detections WHERE sha256 = $1 AND model = $2",
      [sha256, model],
    );
    return rows[0]?.logit;
  }

  async save(sha256: Buffer, model: string, logit: number): Promise<void> {
    await this.pool.query(
      `INSERT INTO detections (sha256, model, logit) VALUES ($1, $2, $3)
       ON CONFLICT (sha256, model) DO NOTHING`,
      [sha256, model, logit],
    );
  }
}
