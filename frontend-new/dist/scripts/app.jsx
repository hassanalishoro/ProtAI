/* ============================================================
   Main app — routes between pages
   ============================================================ */

function App() {
  const route = useHashRoute();

  let page;
  switch (route) {
    case "demo":         page = <PageDemo/>;          break;
    case "architecture": page = <PageArchitecture/>;  break;
    case "methodology":  page = <PageMethodology/>;   break;
    case "results":      page = <PageResults/>;       break;
    case "reference":    page = <PageReference/>;     break;
    case "home":
    default:             page = <PageLanding/>;
  }

  return (
    <>
      <a href="#main" className="skip-link">Skip to content</a>
      <Nav route={route}/>
      <main id="main">
        {page}
      </main>
      <Footer/>
      <ProtAITweaks/>
    </>
  );
}

const root = ReactDOM.createRoot(document.getElementById("app"));
root.render(<App/>);
