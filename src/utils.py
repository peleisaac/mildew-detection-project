import streamlit as st

def display_prediction(label, confidence):
    if label.lower() == "healthy":
        st.markdown(f"""
        <div class="healthy-result">
            <h3>✅ HEALTHY LEAF</h3>
            <h4>Confidence: {confidence:.1f}%</h4>
            <p>No signs of powdery mildew detected</p>
        </div>
        """, unsafe_allow_html=True)

        st.success("**Recommendation**: Continue normal care routine. Monitor regularly for early detection.")

    else:
        st.markdown(f"""
        <div class="mildew-result">
            <h3>⚠️ MILDEW DETECTED</h3>
            <h4>Confidence: {confidence:.1f}%</h4>
            <p>Powdery mildew infection identified</p>
        </div>
        """, unsafe_allow_html=True)

        st.error("**Immediate Action Required**: Apply fungicide treatment immediately. Isolate affected trees to prevent spread.")
