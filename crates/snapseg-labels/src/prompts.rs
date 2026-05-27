//! In-memory ↔ on-disk conversion for snapseg prompt sessions.
//!
//! [`PromptSession`](snapseg_core::PromptSession) is what the app
//! accumulates from user clicks; [`PromptSessionJson`](crate::PromptSessionJson)
//! is the durable shape persisted into `prompts.json`. The conversion
//! is one-way today: snapseg-app writes; downstream training tools
//! read.

use crate::types::{LabelError, PromptRecord, PromptSessionJson};

/// Convert a live [`snapseg_core::PromptSession`] into the on-disk
/// JSON shape.
///
/// `t_ms_offsets` is one entry per prompt in `session.prompts`
/// giving the milliseconds since the first prompt; pass all zeros
/// when the caller didn't track timing.
///
/// # Errors
///
/// Returns [`LabelError::InvalidInput`] if `t_ms_offsets.len()`
/// differs from `session.prompts.len()`.
pub fn prompts_to_json(
    session: &snapseg_core::PromptSession,
    t_ms_offsets: &[u64],
) -> Result<PromptSessionJson, LabelError> {
    if t_ms_offsets.len() != session.prompts.len() {
        return Err(LabelError::InvalidInput(
            "t_ms_offsets length must match session.prompts".to_string(),
        ));
    }
    let mut session_json: Vec<PromptRecord> = Vec::with_capacity(session.prompts.len());
    for (prompt, &t_ms) in session.prompts.iter().zip(t_ms_offsets.iter()) {
        let record = match prompt {
            snapseg_core::Prompt::Click { point, polarity } => PromptRecord::Click {
                x: point.x,
                y: point.y,
                polarity: polarity_str(*polarity).to_string(),
                t_ms,
            },
            snapseg_core::Prompt::Box(b) => PromptRecord::Box {
                x0: b.x0,
                y0: b.y0,
                x1: b.x1,
                y1: b.y1,
                t_ms,
            },
            snapseg_core::Prompt::Scribble { points, polarity } => PromptRecord::Scribble {
                points: points.iter().map(|p| [p.x, p.y]).collect(),
                polarity: polarity_str(*polarity).to_string(),
                t_ms,
            },
        };
        session_json.push(record);
    }
    Ok(PromptSessionJson {
        session: session_json,
    })
}

fn polarity_str(p: snapseg_core::Polarity) -> &'static str {
    match p {
        snapseg_core::Polarity::Positive => "positive",
        snapseg_core::Polarity::Negative => "negative",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use snapseg_core::{BBox, Point2, Polarity, Prompt, PromptSession};

    #[test]
    fn prompts_to_json_roundtrip() {
        let mut s = PromptSession::new();
        s.push(Prompt::Click {
            point: Point2::new(1.0, 2.0),
            polarity: Polarity::Positive,
        });
        s.push(Prompt::Box(BBox {
            x0: 0.0,
            y0: 1.0,
            x1: 4.0,
            y1: 5.0,
        }));
        s.push(Prompt::Scribble {
            points: vec![Point2::new(0.5, 0.5), Point2::new(1.5, 1.5)],
            polarity: Polarity::Negative,
        });
        let offsets = vec![0, 100, 250];
        let json = prompts_to_json(&s, &offsets).expect("prompts_to_json");
        assert_eq!(json.session.len(), 3);

        match &json.session[0] {
            PromptRecord::Click {
                x,
                y,
                polarity,
                t_ms,
            } => {
                assert!((*x - 1.0).abs() < 1e-6);
                assert!((*y - 2.0).abs() < 1e-6);
                assert_eq!(polarity, "positive");
                assert_eq!(*t_ms, 0);
            }
            other => panic!("expected Click variant, got {other:?}"),
        }
        match &json.session[1] {
            PromptRecord::Box {
                x0,
                y0,
                x1,
                y1,
                t_ms,
            } => {
                assert!((*x0 - 0.0).abs() < 1e-6);
                assert!((*y0 - 1.0).abs() < 1e-6);
                assert!((*x1 - 4.0).abs() < 1e-6);
                assert!((*y1 - 5.0).abs() < 1e-6);
                assert_eq!(*t_ms, 100);
            }
            other => panic!("expected Box variant, got {other:?}"),
        }
        match &json.session[2] {
            PromptRecord::Scribble {
                points,
                polarity,
                t_ms,
            } => {
                assert_eq!(points.len(), 2);
                assert!((points[0][0] - 0.5).abs() < 1e-6);
                assert!((points[1][1] - 1.5).abs() < 1e-6);
                assert_eq!(polarity, "negative");
                assert_eq!(*t_ms, 250);
            }
            other => panic!("expected Scribble variant, got {other:?}"),
        }
    }

    #[test]
    fn prompts_to_json_length_mismatch() {
        let mut s = PromptSession::new();
        s.push(Prompt::Click {
            point: Point2::new(0.0, 0.0),
            polarity: Polarity::Positive,
        });
        let err = prompts_to_json(&s, &[]).expect_err("length mismatch should error");
        match err {
            LabelError::InvalidInput(msg) => {
                assert!(msg.contains("t_ms_offsets"), "msg = {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}
