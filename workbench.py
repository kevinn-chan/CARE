import math_engine as risk_engine

def run_scenario(name, road, flow, weather, vision_likelihood):
    print(f"\n{'=' * 60}")
    print(f"🔬WOorkbench Scenario: {name}")
    print(f"{'=' * 60}")
    print(f"Context   : {road} | {flow} | {weather}")
    print(f"Telemetry : Vision P(V|T) = {vision_likelihood:.4f}")
    print("-" * 60)

    #Stochastic Engine
    prior, var_threshold = risk_engine.monte_carlo_sims(
        flow=flow,
        weather=weather,
        road=road
    )

    #Bayesian Fusion
    posterior = risk_engine.calculate_bayesian_posterior(prior, vision_likelihood)

    #calculate Delta (risk spread)
    spread = posterior - var_threshold

    print(f"Contextual Prior P(T) : {prior:.4f}")
    print(f"95% VaR Threshold     : {var_threshold:.4f}")
    print(f"Final Posterior P(T|V): {posterior:.4f}")
    print("-" * 60)

    if posterior > var_threshold:
        print(f"🚨RESULT: BREACH DETECTED (+{spread:.4f} over VaR)")
        print("Logic: Visual evidence outweighed environmental uncertainty.")
    else:
        print(f"✅RESULT: ALARM SUPPRESSED ({spread:.4f} under VaR)")
        print("Logic: Visual anomaly absorbed by contextual noise floor.")


if __name__ == "__main__":
    print("\n🚀INITIALIZING B-SVaR TRAFFIC WORKBENCH...\n")

    #scenario 1 (base line): Normal Day
    #moderate vision score should trigger an alert because baseline is stable
    run_scenario(
        name="Normal Day (base line)",
        road="Straight City Road",
        flow="Normal Flow",
        weather="Clear/Dry",
        vision_likelihood=0.65
    )

    #scenario 2: False Positive Prevention (high noise, but safe vision)
    #higher VaR threshold due to rain and night time
    run_scenario(
        name="False Positive Prevention",
        road="Complex Intersection",
        flow="Night (Low Visibility)",
        weather="Heavy Rain/Wet Asphalt",
        vision_likelihood=0.20
    )

    #scenario 3: False Positive Check (high noise, moderate vision certainty)
    #traffic jam and rain might cause visual confusion, system should aggressively suppress false positives
    run_scenario(
        name="False Positive Check",
        road="High-Speed Highway",
        flow="Rush Hour (Gridlock)",
        weather="Heavy Rain/Wet Asphalt",
        vision_likelihood=0.70
    )

    #scenario 4: Actual Crash (high noise, high vision certainty)
    #definite crash even though VaR threshold is high
    run_scenario(
        name="Actual Crash",
        road="Complex Intersection",
        flow="Rush Hour (Gridlock)",
        weather="Heavy Rain/Wet Asphalt",
        vision_likelihood=0.95
    )

    print(f"\n{'=' * 60}")
    print("🏁Workbench complete. All stochastic boundaries verified.")
    print(f"{'=' * 60}\n")
